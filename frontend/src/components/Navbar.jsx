import React, { useState, useContext, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Menu, X } from "lucide-react";
import logo from "../assets/medsai-logo2-white.png";
import { AuthContext } from "../context/AuthContext";

const Navbar = ({ className }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useContext(AuthContext);
  
  const currentPath = location.pathname;

  const isActive = (path) => {
    if (path === "/") {
      return currentPath === "/" ? "active" : "";
    }
    return currentPath.includes(path) ? "active" : "";
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  // Close menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  return (
    <header className={`navigation-bar ${className || ""}`}>
      <div className="logo-container">
        <img className="logo" src={logo} alt="image" /> 
        <span className="app-name">MEDS-AI</span>
      </div>
      
      {/* Mobile Menu Toggle */}
      <div className="mobile-menu-toggle" onClick={toggleMenu}>
        {isMenuOpen ? <X size={24} color="white" /> : <Menu size={24} color="white" />}
      </div>

      <ul className={`nav-links ${isMenuOpen ? 'mobile-menu-open' : ''}`}>
        <li>
          <Link to="/" className={isActive("/")}>
            Home
          </Link>
        </li>
        
        {isAuthenticated ? (
          // Navigation items for logged-in users
          <>
            <li>
              <Link to="/predict" className={isActive("/predict")}>
                Predict Disease
              </Link>
            </li>
            <li>
              <Link to="/hospital" className={isActive("/hospital")}>
                Hospitals
              </Link>
            </li>
            <li>
              <Link to="/doctors" className={isActive("/doctors")}>
                Doctors
              </Link>
            </li>
            <li>
              <Link to="/previous-predictions" className={isActive("/previous-predictions")}>
                Previous Predictions
              </Link>
            </li>
            <li>
              <Link to="/appointments" className={isActive("/appointments")}>
                Appointments
              </Link>
            </li>
            <li className="text-white">|</li>
            <li>
              <span className="nav-item user-greeting">Hi, {user?.name}</span>
            </li>
            <li>
              <button 
                onClick={handleLogout}
                className="nav-item logout-btn"
              >
                Logout
              </button>
            </li>
          </>
        ) : (
          // Navigation items for guests/not logged-in users
          <>
            <li>
              <Link to="/login" className={isActive("/login")}>
                Login
              </Link>
            </li>
            <li>
              <Link to="/predict" className={isActive("/predict")}>
                Predict Disease
              </Link>
            </li>
            <li>
              <Link to="/hospital" className={isActive("/hospital")}>
                Hospitals
              </Link>
            </li>
            <li>
              <Link to="/aboutus" className={isActive("/about")}>
                About Us
              </Link>
            </li>
            <li>
              <Link to="/contact" className={isActive("/contact")}>
                Contact
              </Link>
            </li>
          </>
        )}
      </ul>
    </header>
  );
};

export default Navbar;