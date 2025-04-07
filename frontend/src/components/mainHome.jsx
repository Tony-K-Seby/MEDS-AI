


import React, { useState } from "react";
import Squares from "./Squares";
import { Link } from "react-router-dom";

const HomePage = () => {
  // State for controlling popups
  const [showLearnMorePopup, setShowLearnMorePopup] = useState(false);
  const [showPrivacyPopup, setShowPrivacyPopup] = useState(false);

  // Function to handle the Learn More button click
  const handleLearnMore = () => {
    setShowLearnMorePopup(true);
  };

  // Function to handle the Privacy Policy link click
  const handlePrivacyPolicy = (e) => {
    e.preventDefault();
    setShowPrivacyPopup(true);
  };

  return (
    <div
      className="homepage-container"
      style={{ position: "relative", overflow: "hidden" }}
    >
      {/* Animated Background - keeping original structure */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          zIndex: -1,
        }}
      >
        <style>
          {`
            /* Metallic styling for squares without changing their layout */
            .metallic-square {
              border-color: #ffc05c !important;
              box-shadow: 0 0 10px rgba(255, 192, 92, 0.3);
              transition: all 0.5s ease !important;
            }
            
            .metallic-square:hover {
              background-color: rgba(255, 192, 92, 0.1) !important;
              transform: scale(1.02) !important;
            }
            
            /* Button styling */
            .cta-btn {
              background: linear-gradient(135deg, #ff9c00 0%, #ff8900 100%);
              color: #222;
              font-weight: bold;
              border: none;
              padding: 12px 24px;
              border-radius: 30px;
              margin: 10px;
              cursor: pointer;
              transition: all 0.3s ease;
              box-shadow: 0 5px 15px rgba(255, 140, 0, 0.3);
            }
            
            .cta-btn:hover {
              transform: translateY(-3px);
              box-shadow: 0 8px 20px rgba(255, 140, 0, 0.4);
            }
            
            /* Feature card styling - keeping original layout */
            .feature-card {
              background:linear-gradient(135deg, rgb(255 255 255) 0%, rgba(255, 140, 0, 0.05) 100%);
              border-radius: 16px;
             
              box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
              backdrop-filter: blur(5px);
              transition: all 0.3s ease;
            }
            
            .feature-card:hover {
              transform: translateY(-5px);
              box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }
            
            /* Footer with original blue box color */
            .homepage-footer {
              background-color: #7fb2e6;
              padding: 20px 0;
            }
            
            .footer-links ul {
              display: flex;
              justify-content: center;
              list-style: none;
              padding: 0;
              margin: 0;
            }
            
            .footer-links a {
              color: #333;
              margin: 0 20px;
              text-decoration: none;
              position: relative;
              transition: all 0.3s ease;
              font-weight: bold;
            }
            
            .footer-links a:after {
              content: '';
              position: absolute;
              width: 0;
              height: 2px;
              bottom: -5px;
              left: 0;
              background-color: #ffc05c;
              transition: width 0.3s ease;
            }
            
            .footer-links a:hover:after {
              width: 100%;
            }
            
            /* Hero section styling */
            .hero-title {
              background: linear-gradient(to right, #ffffff, #ff8c00);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              text-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
            }
            
            /* Popup styling */
            .popup-overlay {
              position: fixed;
              top: 0;
              left: 0;
              right: 0;
              bottom: 0;
              background-color: rgba(0, 0, 0, 0.7);
              display: flex;
              justify-content: center;
              align-items: center;
              z-index: 1000;
              animation: fadeIn 0.3s ease;
            }
            
            .popup-content {
              background: linear-gradient(135deg, #fff 0%, #f8f8f8 100%);
              border-radius: 20px;
              padding: 30px;
              max-width: 600px;
              width: 90%;
              box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
              position: relative;
              animation: slideUp 0.4s ease;
            }
            
            .popup-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 20px;
              border-bottom: 2px solid #ffc05c;
              padding-bottom: 10px;
            }
            
            .popup-title {
              font-size: 24px;
              color: #333;
              margin: 0;
            }
            
            .popup-close {
              background: linear-gradient(135deg, #ff8c00 0%, #ffc05c 100%);
              color: white;
              border: none;
              width: 30px;
              height: 30px;
              border-radius: 50%;
              display: flex;
              justify-content: center;
              align-items: center;
              cursor: pointer;
              font-weight: bold;
              box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
              transition: all 0.3s ease;
            }
            
            .popup-close:hover {
              transform: scale(1.1);
              box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3);
            }
            
            .popup-body {
              font-size: 16px;
              line-height: 1.6;
              color: #444;
            }
            
            .popup-footer {
              margin-top: 20px;
              display: flex;
              justify-content: flex-end;
            }
            
            .popup-footer button {
              background: linear-gradient(135deg, #ffc05c 0%, #ff8c00 100%);
              color: #222;
              font-weight: bold;
              border: none;
              padding: 10px 20px;
              border-radius: 30px;
              cursor: pointer;
              transition: all 0.3s ease;
              box-shadow: 0 3px 8px rgba(255, 140, 0, 0.3);
            }
            
            .popup-footer button:hover {
              transform: translateY(-3px);
              box-shadow: 0 5px 15px rgba(255, 140, 0, 0.4);
            }
            
            @keyframes fadeIn {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            
            @keyframes slideUp {
              from { transform: translateY(50px); opacity: 0; }
              to { transform: translateY(0); opacity: 1; }
            }
            
            /* Funny animation for learn more popup */
            .learn-more-content {
              text-align: center;
            }
            
            .congratulations {
              font-size: 28px;
              font-weight: bold;
              margin: 20px 0;
              color: #ff8c00;
              animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
              0% { transform: scale(1); }
              50% { transform: scale(1.1); }
              100% { transform: scale(1); }
            }
            
            .privacy-content {
              max-height: 400px;
              overflow-y: auto;
              padding-right: 10px;
            }
            
            .privacy-content h3 {
              color: #ff8c00;
              margin-top: 20px;
              margin-bottom: 10px;
            }
            
            .privacy-content::-webkit-scrollbar {
              width: 8px;
            }
            
            .privacy-content::-webkit-scrollbar-track {
              background: #f1f1f1;
              border-radius: 10px;
            }
            
            .privacy-content::-webkit-scrollbar-thumb {
              background: #ffc05c;
              border-radius: 10px;
            }
          `}
        </style>
        <Squares
          speed={0.2}
          squareSize={40}
          direction="diagonal" // maintaining original settings
          borderColor="#ffc05c"
          hoverFillColor="#222"
          className="metallic-square"
        />
      </div>

      {/* Navbar Component */}
      {/* <Navbar /> */}
      <br />

      {/* Hero Section - keeping original structure */}
      <section className="hero-section">
        <h2 className="hero-title">Your Health, Powered by AI</h2>
        <p className="hero-description">
          Start predicting diseases with our advanced AI models and take charge
          of your health.
        </p>
        <div className="cta-buttons">
          <Link to="/predict"><button className="cta-btn">Start Prediction</button></Link>
          
          <button className="cta-btn" onClick={handleLearnMore}>Learn More</button>
        </div>
        {/* <img className="hero-image" src={healthApp} alt="image" /> */}
      </section>

      {/* Features Section - keeping original layout */}
      <section className="features-section">
        <div className="feature-card">
          <h3 className="feature-title">Disease Prediction using AI</h3>
          <p className="feature-description">
            Our advanced machine learning models analyze symptoms and predict
            potential diseases with high accuracy.
          </p>
        </div>
        <div className="feature-card">
          <h3 className="feature-title">Personalized Doctor Recommendations</h3>
          <p className="feature-description">
            Get matched with doctors based on location, specialty, and
            availability for personalized consultations.
          </p>
        </div>
        <div className="feature-card">
          <h3 className="feature-title">Verified Home Remedies Suggestions</h3>
          <p className="feature-description">
            Discover expert-reviewed home remedies for common conditions and
            start improving your health naturally.
          </p>
        </div>
        <div className="feature-card">
          <h3 className="feature-title">User-Friendly Reports</h3>
          <p className="feature-description">
            Receive clear and actionable health reports that help you make
            better health decisions.
          </p>
        </div>
      </section>

      {/* Footer with original blue box */}
      <footer className="homepage-footer">
        <div className="footer-links">
          <ul>
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/aboutus">About Us</Link>
            </li>
            <li>
              <a href="#privacy" onClick={handlePrivacyPolicy}>Privacy Policy</a>
            </li>
          </ul>
        </div>
      </footer>

      {/* Learn More Popup */}
      {showLearnMorePopup && (
        <div className="popup-overlay" onClick={() => setShowLearnMorePopup(false)}>
          <div className="popup-content" onClick={(e) => e.stopPropagation()}>
            <div className="popup-header">
              <h3 className="popup-title">Learning Complete!</h3>
              <button className="popup-close" onClick={() => setShowLearnMorePopup(false)}>×</button>
            </div>
            <div className="popup-body learn-more-content">
              <div className="congratulations">Congratulations! You've learned more! 🎉</div>
              <p>Your brain has officially expanded by approximately 0.0001%</p>
              <p>Side effects may include: feeling smarter, urge to predict diseases, and sudden interest in AI technologies.</p>
              <img src="https://media.giphy.com/media/d3mlE7uhX8KFgEmY/source.gif" alt="Smart GIF" style={{ width: '80%', borderRadius: '10px', marginTop: '15px' }} />
            </div>
            <div className="popup-footer">
              <button onClick={() => setShowLearnMorePopup(false)}>Close (You're Smarter Now)</button>
            </div>
          </div>
        </div>
      )}

      {/* Privacy Policy Popup */}
      {showPrivacyPopup && (
        <div className="popup-overlay" onClick={() => setShowPrivacyPopup(false)}>
          <div className="popup-content" onClick={(e) => e.stopPropagation()}>
            <div className="popup-header">
              <h3 className="popup-title">Privacy Policy</h3>
              <button className="popup-close" onClick={() => setShowPrivacyPopup(false)}>×</button>
            </div>
            <div className="popup-body privacy-content">
              <p>Last Updated: April 3, 2025</p>
              
              <h3>Introduction</h3>
              <p>Welcome to HealthAI. We respect your privacy and are committed to protecting your personal health information. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our disease prediction service.</p>
              
              <h3>Information We Collect</h3>
              <p>We collect the following types of information:</p>
              <p><strong>Personal Information:</strong> Name, email address, age, gender, and contact details.</p>
              <p><strong>Health Information:</strong> Symptoms, medical history, and other health-related data you provide for prediction purposes.</p>
              <p><strong>Usage Data:</strong> Information about how you interact with our application, including features used and time spent.</p>
              
              <h3>How We Use Your Information</h3>
              <p>We use your information to:</p>
              <p>• Provide accurate disease predictions based on your symptoms</p>
              <p>• Recommend appropriate healthcare professionals</p>
              <p>• Improve our AI models and prediction algorithms</p>
              <p>• Communicate important updates about our service</p>
              <p>• Ensure the security and functionality of our platform</p>
              
              <h3>Data Security</h3>
              <p>We implement strict security measures to protect your personal and health information. This includes encryption, secure servers, regular security audits, and strict access controls for our staff.</p>
              
              <h3>Sharing Your Information</h3>
              <p>We do not sell your personal information. We may share anonymized data with:</p>
              <p>• Healthcare professionals you choose to connect with</p>
              <p>• Research partners (with anonymized data only)</p>
              <p>• Service providers who help us operate our platform</p>
              
              <h3>Your Rights</h3>
              <p>You have the right to:</p>
              <p>• Access your personal information</p>
              <p>• Correct inaccurate information</p>
              <p>• Delete your account and associated data</p>
              <p>• Object to certain processing of your data</p>
              <p>• Export your data in a portable format</p>
              
              <h3>Important Medical Disclaimer</h3>
              <p>The disease predictions provided are for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider regarding any health concerns.</p>
              
              <h3>Contact Us</h3>
              <p>If you have questions about this Privacy Policy, please contact our Data Protection Officer at privacy@healthai.example.com</p>
            </div>
            <div className="popup-footer">
              <button onClick={() => setShowPrivacyPopup(false)}>I Understand</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HomePage;