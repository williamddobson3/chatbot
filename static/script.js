// Chatbot frontend JavaScript

const API_BASE = '/api';
let conversationHistory = [];

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const status = document.getElementById('status');

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Send message on Enter (Shift+Enter for new line)
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send button click
sendButton.addEventListener('click', sendMessage);

// Clear chat
clearButton.addEventListener('click', clearChat);

// Check API health on load
checkHealth();

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            updateStatus('Ready', 'success');
        } else {
            updateStatus('Model loading...', 'connecting');
        }
    } catch (error) {
        updateStatus('Connection error', 'error');
        console.error('Health check failed:', error);
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // Disable input
    userInput.disabled = true;
    sendButton.disabled = true;
    updateStatus('Thinking...', 'connecting');
    
    // Add user message to UI
    addMessage('user', message);
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Add user message to history
    conversationHistory.push({
        role: 'user',
        content: message
    });
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Send request to API
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: conversationHistory
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add bot response to UI
        addMessage('bot', data.response);
        
        // Add bot response to history
        conversationHistory.push({
            role: 'assistant',
            content: data.response
        });
        
        updateStatus('Ready', 'success');
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('bot', `Sorry, I encountered an error: ${error.message}. Please try again.`);
        updateStatus('Error occurred', 'error');
        console.error('Error:', error);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const p = document.createElement('p');
    // Simple markdown-like formatting
    p.textContent = content;
    contentDiv.appendChild(p);
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const dots = document.createElement('div');
    dots.className = 'typing-indicator';
    dots.innerHTML = '<span></span><span></span><span></span>';
    
    contentDiv.appendChild(dots);
    typingDiv.appendChild(contentDiv);
    chatMessages.appendChild(typingDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'typing-indicator';
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        conversationHistory = [];
        chatMessages.innerHTML = `
            <div class="message bot-message">
                <div class="message-content">
                    <p>Chat cleared. How can I help you?</p>
                </div>
            </div>
        `;
        updateStatus('Ready', 'success');
    }
}

function updateStatus(text, type = '') {
    status.textContent = text;
    status.className = `status ${type}`;
}

