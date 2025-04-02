import React from "react";
import Squares from "./Squares";
import { Link } from "react-router-dom";


const HomePage = () => {
  return (
    <div
      className="homepage-container"
      style={{ position: "relative", overflow: "hidden" }}
    >
      {/* Animated Background */}
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
        <Squares
          speed={0.2}
          squareSize={40}
          direction="diagonal" // up, down, left, right, diagonal
          borderColor="#ffc05c"
          hoverFillColor="#222"
        />
      </div>

      {/* Navbar Component */}
      {/* <Navbar /> */}
      <br />

      {/* Hero Section */}
      <section className="hero-section">
        <h2 className="hero-title">Your Health, Powered by AI</h2>
        <p className="hero-description">
          Start predicting diseases with our advanced AI models and take charge
          of your health.
        </p>
        <div className="cta-buttons">
          <Link to="/predict"><button className="cta-btn">Start Prediction</button></Link>
          
          <button className="cta-btn">Learn More</button>
        </div>
        {/* <img className="hero-image" src={healthApp} alt="image" /> */}
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
      </footer>
    </div>
  );
};

export default HomePage;
