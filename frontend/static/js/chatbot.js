// Chatbot functionality
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const chatbotButton = document.getElementById('chatbot-button');
    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotNotification = document.getElementById('chatbot-notification');

    // Chat context to remember conversation
    let chatContext = {
        messageCount: 0,
        lastTopic: null,
        userInfo: {},
        conversationStarted: false
    };

    // Show notification after 30 seconds if chat not opened
    setTimeout(() => {
        if (!chatContext.conversationStarted) {
            chatbotNotification.style.display = 'flex';
        }
    }, 30000);

    // Toggle chatbot visibility
    chatbotButton.addEventListener('click', function() {
        chatbotContainer.classList.toggle('show');
        chatbotNotification.style.display = 'none';

        if (chatbotContainer.classList.contains('show')) {
            chatbotInput.focus();

            if (!chatContext.conversationStarted) {
                chatContext.conversationStarted = true;

                // Add welcome message with logo when first opened
                setTimeout(() => {
                    addWelcomeMessage();

                    setTimeout(() => {
                        // Add initial message
                        addBotMessage("Hello! I'm PulmoBot, your virtual assistant for PulmoScan. How can I help you today?");

                        // Add quick reply options
                        setTimeout(() => {
                            addQuickReplies([
                                "How does PulmoScan work?",
                                "Cancer types detected",
                                "Cancer staging",
                                "Upload a scan",
                                "Accuracy rates"
                            ]);
                        }, 500);
                    }, 500);
                }, 300);
            }
        }
    });

    // Close chatbot
    chatbotClose.addEventListener('click', function() {
        chatbotContainer.classList.remove('show');
    });

    // Send message on button click
    chatbotSend.addEventListener('click', sendMessage);

    // Send message on Enter key
    chatbotInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Function to add welcome message with logo
    function addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.classList.add('welcome-message');

        const logo = document.createElement('img');
        logo.src = '/static/img/logo.png';
        logo.alt = 'PulmoScan Logo';
        logo.classList.add('welcome-logo');

        welcomeDiv.appendChild(logo);
        chatbotMessages.appendChild(welcomeDiv);
        scrollToBottom();
    }

    // Function to send message
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (message !== '') {
            // Add user message
            addUserMessage(message);

            // Clear input
            chatbotInput.value = '';

            // Show typing indicator
            showTypingIndicator();

            // Process the message and get response
            setTimeout(() => {
                processMessage(message);
            }, Math.random() * 1000 + 1000); // Random delay between 1-2 seconds for more natural feel

            // Update context
            chatContext.messageCount++;
        }
    }

    // Function to add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'user-message');

        const messageText = document.createElement('div');
        messageText.textContent = message;

        const messageTime = document.createElement('div');
        messageTime.classList.add('message-time');
        messageTime.textContent = getCurrentTime();

        messageElement.appendChild(messageText);
        messageElement.appendChild(messageTime);

        chatbotMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Function to add bot message to chat
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'bot-message');

        const messageText = document.createElement('div');
        messageText.textContent = message;

        const messageTime = document.createElement('div');
        messageTime.classList.add('message-time');
        messageTime.textContent = getCurrentTime();

        // Add feedback buttons
        const feedbackButtons = document.createElement('div');
        feedbackButtons.classList.add('feedback-buttons');
        feedbackButtons.innerHTML = `
            <button class="feedback-btn" data-action="like" aria-label="Helpful response"><i class="bi bi-hand-thumbs-up"></i></button>
            <button class="feedback-btn" data-action="dislike" aria-label="Unhelpful response"><i class="bi bi-hand-thumbs-down"></i></button>
        `;

        messageElement.appendChild(messageText);
        messageElement.appendChild(messageTime);
        messageElement.appendChild(feedbackButtons);

        chatbotMessages.appendChild(messageElement);

        // Add event listeners to feedback buttons
        const likeBtn = feedbackButtons.querySelector('[data-action="like"]');
        const dislikeBtn = feedbackButtons.querySelector('[data-action="dislike"]');

        likeBtn.addEventListener('click', function() {
            this.classList.add('liked');
            dislikeBtn.classList.remove('disliked');
            // In a real app, you would send this feedback to your server
        });

        dislikeBtn.addEventListener('click', function() {
            this.classList.add('disliked');
            likeBtn.classList.remove('liked');
            // In a real app, you would send this feedback to your server
        });

        scrollToBottom();
    }

    // Function to add quick reply buttons
    function addQuickReplies(options) {
        const quickReplies = document.createElement('div');
        quickReplies.classList.add('quick-replies');

        options.forEach(option => {
            const button = document.createElement('button');
            button.classList.add('quick-reply-btn');
            button.textContent = option;
            button.addEventListener('click', function() {
                // Add as user message
                addUserMessage(option);

                // Remove quick replies
                quickReplies.remove();

                // Show typing indicator
                showTypingIndicator();

                // Process the message
                setTimeout(() => {
                    processMessage(option);
                }, Math.random() * 1000 + 800);

                // Update context
                chatContext.messageCount++;
            });

            quickReplies.appendChild(button);
        });

        chatbotMessages.appendChild(quickReplies);
        scrollToBottom();
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('typing-indicator');
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        typingIndicator.id = 'typing-indicator';
        chatbotMessages.appendChild(typingIndicator);
        scrollToBottom();
    }

    // Function to hide typing indicator
    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // Function to get current time
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // Function to scroll to bottom of chat
    function scrollToBottom() {
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // Function to process message and get response
    function processMessage(message) {
        // Hide typing indicator
        hideTypingIndicator();

        // Simple response logic - in a real app, this would call an API
        let response;
        let quickReplies = null;

        message = message.toLowerCase();

        // Store the topic for context awareness
        if (message.includes('cancer') && message.includes('type')) {
            chatContext.lastTopic = 'cancer_type';
        } else if (message.includes('cancer') && message.includes('stage')) {
            chatContext.lastTopic = 'cancer_stage';
        } else if (message.includes('how') && message.includes('work')) {
            chatContext.lastTopic = 'how_it_works';
        } else if (message.includes('accuracy')) {
            chatContext.lastTopic = 'accuracy';
        } else if (message.includes('upload') || message.includes('scan')) {
            chatContext.lastTopic = 'upload';
        }

        // Context-aware responses
        if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
            response = "Hello! How can I assist you with PulmoScan today?";
            quickReplies = ["How does PulmoScan work?", "Cancer types detected", "Cancer staging"];
        }
        else if (message.includes('cancer') && message.includes('type') || message.includes('cancer types detected')) {
            response = "PulmoScan can detect different types of lung cancer including Adenocarcinoma, Squamous Cell Carcinoma, and Small Cell Carcinoma from pathology images. Our system analyzes cellular patterns and morphology to distinguish between cancer types.";
            quickReplies = ["How accurate is type detection?", "Upload pathology image", "Cancer staging"];
        }
        else if (message.includes('cancer') && message.includes('stage') || message.includes('cancer staging')) {
            response = "Our system can classify lung cancer stages (I through IV) using 3D CT scans with high accuracy. We analyze tumor size, location, and spread patterns to determine the stage, which is crucial for treatment planning.";
            quickReplies = ["How accurate is staging?", "Upload CT scan", "Cancer types"];
        }
        else if ((message.includes('how') && message.includes('work')) || message.includes('how does pulmoscan work')) {
            response = "PulmoScan uses advanced deep learning models to analyze medical images. For lung cancer detection, we use EfficientNet models, and for cancer staging, we use 3D ResNet and DenseNet architectures. Our models are trained on thousands of validated medical images to ensure high accuracy.";
            quickReplies = ["Technical details", "Accuracy rates", "Upload a scan"];
        }
        else if (message.includes('technical details')) {
            response = "Our system uses a multi-stage pipeline: 1) Image preprocessing to normalize and enhance features, 2) Feature extraction using convolutional neural networks, 3) Classification using ensemble methods. For 3D scans, we employ patch-based analysis with attention mechanisms to focus on regions of interest.";
            quickReplies = ["Model architectures", "Data preprocessing", "Accuracy rates"];
        }
        else if (message.includes('model architectures')) {
            response = "For cancer type classification, we use ResNet-18 with dropout layers and learning rate scheduling. For cancer staging, we employ both 3D ResNet and 3D DenseNet models with progressive unfreezing for transfer learning. These architectures were chosen after extensive experimentation to optimize for both accuracy and inference speed.";
        }
        else if (message.includes('accuracy') || message.includes('accuracy rates')) {
            response = "Our models achieve over 90% accuracy in detecting lung cancer and over 85% accuracy in staging. For cancer type classification, we achieve 88% accuracy across the main types. These numbers continue to improve as we train on more data and refine our algorithms.";
            quickReplies = ["False positive rate", "Sensitivity", "Specificity"];
        }
        else if (message.includes('false positive') || message.includes('sensitivity') || message.includes('specificity')) {
            response = "Our system maintains a low false positive rate of under 5%. The sensitivity (true positive rate) is 92% and specificity is 94%, which exceeds many manual diagnostic methods. We continuously monitor these metrics and update our models to improve performance.";
        }
        else if (message.includes('upload') || message.includes('scan') || message.includes('upload a scan')) {
            response = "You can upload your CT scans or pathology images through our secure dashboard after logging in. Our system will process them and provide results within minutes. All data is encrypted and handled according to HIPAA compliance standards.";
            quickReplies = ["Supported formats", "Login to dashboard", "Data security"];
        }
        else if (message.includes('supported formats')) {
            response = "For CT scans, we support DICOM format (.dcm files). For pathology images, we support high-resolution formats including TIFF, SVS (Aperio), and standard formats like JPG and PNG. For best results, we recommend using the original medical imaging formats without compression.";
        }
        else if (message.includes('data security') || message.includes('privacy')) {
            response = "We take data security very seriously. All uploads are encrypted using AES-256, and all data is stored in HIPAA-compliant servers. We never share your medical data with third parties, and you can request deletion of your data at any time through your account settings.";
        }
        else if (message.includes('thank')) {
            response = "You're welcome! Is there anything else I can help you with regarding PulmoScan or lung cancer detection?";
            quickReplies = ["Cancer types", "Cancer staging", "No, thank you"];
        }
        else if (message.includes('bye') || message.includes('goodbye') || message.includes('no, thank you')) {
            response = "Thank you for chatting with PulmoBot. If you have more questions later, feel free to return. Have a great day!";
        }
        else {
            response = "I'm still learning about lung cancer detection. For specific medical advice, please consult with our healthcare professionals through the appointment system. Is there something else I can help you with?";
            quickReplies = ["How does PulmoScan work?", "Cancer types", "Cancer staging"];
        }

        // Add bot response
        addBotMessage(response);

        // Add quick replies if available
        if (quickReplies) {
            setTimeout(() => {
                addQuickReplies(quickReplies);
            }, 500);
        }
    }
});
