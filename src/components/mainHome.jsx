import React from "react";
import healthApp from "../assets/health.png";

const HomePage = () => {
  return (
    <div className="homepage-container">
      {/* Header */}
      <header class="navigation-bar">
        <div class="logo-container">
          <span class="app-name">MEDS-AI</span>
        </div>
        <ul class="nav-links">
          <li>
            <a href="#home" class="active">
              Home
            </a>
          </li>
          <li>
            <a href="#remedies">Login</a>
          </li>

          <li>
            <a href="#predict">Predict Disease</a>
          </li>
          <li>
            <a href="#doctors">Doctors</a>
          </li>
          <li>
            <a href="#about">About Us</a>
          </li>
          <li>
            <a href="#contact">Contact</a>
          </li>
        </ul>
      </header>
      <br></br>

      {/* Hero Section */}
      <section className="hero-section">
        <h2 className="hero-title">Your Health, Powered by AI</h2>
        <p className="hero-description">
          Start predicting diseases with our advanced AI models and take charge
          of your health.
        </p>
        <div className="cta-buttons">
          <button className="cta-btn">Start Prediction</button>
          <button className="cta-btn">Learn More</button>
        </div>
        <img className="hero-image" src={healthApp} alt="image" />
      </section>

      {/* Features Section */}
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

      {/* Footer */}
      <footer className="homepage-footer">
        <div className="footer-links">
          <ul>
            <li>
              <a href="#home">Home</a>
            </li>
            <li>
              <a href="#about">About Us</a>
            </li>
            <li>
              <a href="#privacy">Privacy Policy</a>
            </li>
          </ul>
        </div>
        <div className="contact-info">
          <p>
            Follow Us: <a href="#">Twitter</a>, <a href="#">Facebook</a>
          </p>
        </div>
        <div className="download-links">
          <a href="#app-store" className="download-btn">
            Download on the App Store
          </a>
          <a href="#google-play" className="download-btn">
            Get it on Google Play
          </a>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
