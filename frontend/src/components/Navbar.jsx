// import React, { useState, useContext, useEffect } from "react";
// import { Link, useLocation, useNavigate } from "react-router-dom";
// import { Menu, X } from "lucide-react";
// import logo from "../assets/medsai-logo2-white.png";
// import { AuthContext } from "../context/AuthContext";

// const Navbar = ({ className }) => {
//   const [isMenuOpen, setIsMenuOpen] = useState(false);
//   const location = useLocation();
//   const navigate = useNavigate();
//   const { isAuthenticated, user, logout } = useContext(AuthContext);
  
//   const currentPath = location.pathname;

//   const isActive = (path) => {
//     if (path === "/") {
//       return currentPath === "/" ? "active" : "";
//     }
//     return currentPath.includes(path) ? "active" : "";
//   };

//   const handleLogout = () => {
//     logout();
//     navigate("/login");
//   };

//   const toggleMenu = () => {
//     setIsMenuOpen(!isMenuOpen);
//   };

//   // Close menu when route changes
//   useEffect(() => {
//     setIsMenuOpen(false);
//   }, [location.pathname]);

//   return (
//     <header className={`navigation-bar ${className || ""}`}>
//       <div className="logo-container">
//         <img className="logo" src={logo} alt="image" /> 
//         <span className="app-name">MEDS-AI</span>
//       </div>
      
//       {/* Mobile Menu Toggle */}
//       <div className="mobile-menu-toggle" onClick={toggleMenu}>
//         {isMenuOpen ? <X size={24} color="white" /> : <Menu size={24} color="white" />}
//       </div>

//       <ul className={`nav-links ${isMenuOpen ? 'mobile-menu-open' : ''}`}>
//         <li>
//           <Link to="/" className={isActive("/")}>
//             Home
//           </Link>
//         </li>
        
//         {isAuthenticated ? (
//           // Navigation items for logged-in users
//           <>
//             <li>
//               <Link to="/predict" className={isActive("/predict")}>
//                 Predict Disease
//               </Link>
//             </li>
//             <li>
//               <Link to="/hospital" className={isActive("/hospital")}>
//                 Hospitals
//               </Link>
//             </li>
//             <li>
//               <Link to="/doctors" className={isActive("/doctors")}>
//                 Doctors
//               </Link>
//             </li>
//             <li>
//               <Link to="/previous-predictions" className={isActive("/previous-predictions")}>
//                 Previous Predictions
//               </Link>
//             </li>
//             <li>
//               <Link to="/appointments" className={isActive("/appointments")}>
//                 Appointments
//               </Link>
//             </li>
//             <li className="text-white">|</li>
//             <li>
//               <span className="nav-item user-greeting">Hi, {user?.name}</span>
//             </li>
//             <li>
//               <button 
//                 onClick={handleLogout}
//                 className="nav-item logout-btn"
//               >
//                 Logout
//               </button>
//             </li>
//           </>
//         ) : (
//           // Navigation items for guests/not logged-in users
//           <>
//             <li>
//               <Link to="/login" className={isActive("/login")}>
//                 Login
//               </Link>
//             </li>
//             <li>
//               <Link to="/predict" className={isActive("/predict")}>
//                 Predict Disease
//               </Link>
//             </li>
//             <li>
//               <Link to="/hospital" className={isActive("/hospital")}>
//                 Hospitals
//               </Link>
//             </li>
//             <li>
//               <Link to="/aboutus" className={isActive("/about")}>
//                 About Us
//               </Link>
//             </li>
//             <li>
//               <Link to="/contact" className={isActive("/contact")}>
//                 Contact
//               </Link>
//             </li>
//           </>
//         )}
//       </ul>
//     </header>
//   );
// };

// export default Navbar;

import React, { useState, useContext, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Menu, X } from "lucide-react";
import logo from "../assets/medsai-logo2-white.png";
import { AuthContext } from "../context/AuthContext";
import TheClockComponent from "./TheClockComponent";

const Navbar = ({ className }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showClock, setShowClock] = useState(false);
  const [logoProgress, setLogoProgress] = useState(0);
  const [isHoldingLogo, setIsHoldingLogo] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useContext(AuthContext);
  const logoRef = useRef(null);
  const timerRef = useRef(null);
  const pressDurationRef = useRef(0);
  
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

  // Handle long-press event on the logo
  const handleLogoMouseDown = () => {
    pressDurationRef.current = 0;
    setIsHoldingLogo(true);
    timerRef.current = setInterval(() => {
      pressDurationRef.current += 100;
      setLogoProgress(Math.min(100, (pressDurationRef.current / 5000) * 100));
      
      if (pressDurationRef.current >= 5000) { // 5 seconds
        clearInterval(timerRef.current);
        setShowClock(true);
        setLogoProgress(0);
        setIsHoldingLogo(false);
      }
    }, 100);
  };

  const handleLogoMouseUp = () => {
    clearInterval(timerRef.current);
    setLogoProgress(0);
    setIsHoldingLogo(false);
  };

  const handleLogoTouchStart = (e) => {
    handleLogoMouseDown();
  };

  const handleLogoTouchEnd = (e) => {
    handleLogoMouseUp();
  };

  // Close menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location.pathname]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // CSS for the logo animation
  const logoAnimationStyle = {
    position: 'relative',
  };

  const logoProgressBarStyle = {
    position: 'absolute',
    bottom: '-5px',
    left: '0',
    height: '3px',
    width: `${logoProgress}%`,
    backgroundColor: isHoldingLogo ? '#ff3333' : 'transparent',
    borderRadius: '2px',
    transition: 'width 0.1s linear'
  };

  const logoContainerClass = `logo-container ${isHoldingLogo ? 'pulse-effect' : ''}`;

  // CSS for the pulse animation to be added to your CSS file
  const pulseStyle = isHoldingLogo ? {
    animation: 'pulse 1s infinite',
    transform: `scale(${1 + (logoProgress * 0.001)})`,
    filter: `brightness(${100 + logoProgress * 0.5}%)`,
    transition: 'transform 0.1s, filter 0.1s',
  } : {};

  return (
    <>
      <header className={`navigation-bar ${className || ""}`}>
        <div 
          className={logoContainerClass}
          ref={logoRef}
          onMouseDown={handleLogoMouseDown}
          onMouseUp={handleLogoMouseUp}
          onMouseLeave={handleLogoMouseUp}
          onTouchStart={handleLogoTouchStart}
          onTouchEnd={handleLogoTouchEnd}
          onTouchCancel={handleLogoTouchEnd}
          style={logoAnimationStyle}
        >
          <img 
            className="logo" 
            src={logo} 
            alt="image" 
            style={pulseStyle}
          /> 
          <span 
            className="app-name"
            style={pulseStyle}
          >
            MEDS-AI
            {isHoldingLogo && 
              <div className="secret-text" style={{
                position: 'absolute',
                top: '100%',
                left: '0',
                fontSize: '10px',
                color: 'rgba(255,51,51,0.8)',
                whiteSpace: 'nowrap',
                opacity: logoProgress / 100
              }}>
                Hold to reveal your fate...
              </div>
            }
          </span>
          <div style={logoProgressBarStyle}></div>
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
      
      {/* The Death Clock Modal */}
      <TheClockComponent 
        isOpen={showClock} 
        onClose={() => setShowClock(false)} 
      />
    </>
  );
};

export default Navbar;