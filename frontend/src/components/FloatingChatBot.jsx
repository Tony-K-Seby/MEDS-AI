import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Send, Loader, RefreshCw, MessageCircle, X } from "lucide-react";

const BASE_URL = import.meta.env.VITE_BASE_URL;


const FloatingChatBot = () => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    startChat();
    
    // Close chat when clicking outside
    const handleClickOutside = (event) => {
      if (chatRef.current && !chatRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const startChat = async () => {
    const newSessionId = Date.now().toString();
    setSessionId(newSessionId);
    setIsLoading(true);

    try {
      const response = await axios.post(`${BASE_URL}/prediction/start_chat`, { session_id: newSessionId });
      setMessages([{ sender: "bot", text: response.data.message }]);
    } catch (error) {
      console.error("Error starting chat:", error);
      setMessages([{ 
        sender: "bot", 
        text: "Sorry, I couldn't connect. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${BASE_URL}/prediction/process_message`, {
        session_id: sessionId,
        user_message: userMessage
      });

      setMessages((prev) => [...prev, { sender: "bot", text: response.data.message }]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev, 
        { sender: "bot", text: "Sorry, something went wrong. Please try again." }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div ref={chatRef} className="fixed bottom-6 right-6 z-50">
      {!isOpen ? (
        // Chat Icon Button
        <button 
          onClick={toggleChat} 
          className="bg-blue-600 hover:bg-blue-700 text-white rounded-full p-4 shadow-lg flex items-center justify-center transition-all duration-300 animate-pulse"
          aria-label="Open chat"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      ) : (
        // Chat Window
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg shadow-xl border border-white border-opacity-20 w-80 md:w-96 flex flex-col transition-all duration-300">
          {/* Chat Header */}
          <div className="flex justify-between items-center p-3 border-b border-white border-opacity-20 bg-blue-600 rounded-t-lg">
            <h2 className="text-lg font-semibold text-white">🩺 Health Assistant</h2>
            <button 
              onClick={toggleChat}
              className="text-white hover:text-gray-200 transition-colors"
              aria-label="Close chat"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          {/* Chat Messages */}
          <div className="h-80 overflow-y-auto p-3 space-y-3 bg-black bg-opacity-40">
            {messages.map((msg, index) => (
              <div 
                key={index} 
                className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div 
                  className={`max-w-xs px-3 py-2 rounded-lg ${
                    msg.sender === "user" 
                      ? "bg-blue-600 text-white rounded-br-none" 
                      : "bg-gray-100 bg-opacity-80 text-gray-800 rounded-bl-none"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 bg-opacity-80 px-3 py-2 rounded-lg rounded-bl-none flex items-center">
                  <Loader className="w-4 h-4 mr-2 animate-spin" />
                  Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          {/* Input area - Increased opacity */}
          <div className="p-3 border-t border-white border-opacity-20 bg-white bg-opacity-20">
            <div className="flex">
              <input
                type="text"

                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Describe your symptoms..."
                className="flex-grow px-3 py-2 bg-white bg-opacity-40 rounded-l-lg focus:outline-none text-black placeholder-gray-200"
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-r-lg transition-colors duration-200"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            
            {/* Reset button - Increased opacity */}
            <button
              onClick={startChat}
              className="flex items-center px-3 py-1 mt-2 bg-white bg-opacity-30 hover:bg-opacity-40 rounded text-white text-xs transition-colors duration-200 w-full justify-center"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Reset Conversation
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FloatingChatBot;