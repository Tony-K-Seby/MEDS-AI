import React, { useState, useEffect, useRef, useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import axios from "axios";

const TheClockComponent = ({ isOpen, onClose }) => {
  const { user } = useContext(AuthContext);
  const [isLoading, setIsLoading] = useState(true);
  const [deathData, setDeathData] = useState(null);
  const [formData, setFormData] = useState({
    dob_day: "1",
    dob_month: "1",
    dob_year: new Date().getFullYear() - 30,
    sex: "Male",
    smoker: false,
    bmi: "<25",
    outlook: "",
    alcohol: "",
    country: "",
    include_fitness_diet: true,
    fitness_level: "",
    diet_rating: ""
  });
  const [countdown, setCountdown] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });
  const [holdAnimation, setHoldAnimation] = useState(false);
  const [holdTimer, setHoldTimer] = useState(null);
  const countdownRef = useRef(null);
  const [debugInfo, setDebugInfo] = useState(null); // Added for debugging

  useEffect(() => {
    if (isOpen && user) {
      checkExistingData();
    }
    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
      if (holdTimer) {
        clearTimeout(holdTimer);
      }
    };
  }, [isOpen, user]);

  useEffect(() => {
    if (deathData && deathData.death_date) {
      startCountdown();
    }
  }, [deathData]);

  const checkExistingData = async () => {
    setIsLoading(true);
    try {
      console.log("Checking for existing data for:", user.email);
      const response = await axios.get(`http://127.0.0.1:5000/theclock/${user.email}`, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      console.log("GET response:", response);
      if (response.status === 200) {
        setDeathData(response.data);
      }
    } catch (error) {
      console.log("No existing death date found:", error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value
    });
  };

  const calculateDeathDate = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setDebugInfo(null);
    
    try {
      const payload = {
        ...formData,
        email: user.email
      };
      
      console.log("Submitting payload:", payload);
      
      // Set up axios with proper CORS headers
      const response = await axios.post("http://127.0.0.1:5000/theclock/", payload, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
      
      console.log("POST response:", response);
      
      if (response.status === 200) {
        setDeathData(response.data);
      }
    } catch (error) {
      console.error("Error calculating death date:", error);
      // Add detailed error info for debugging
      setDebugInfo({
        message: error.message,
        response: error.response ? {
          status: error.response.status,
          data: error.response.data
        } : 'No response data',
        request: error.request ? 'Request was made but no response received' : 'Request setup error'
      });
      
      if (error.response && error.response.data && error.response.data.error) {
        alert(error.response.data.error);
      } else {
        alert("Failed to calculate your death date. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const startCountdown = () => {
    if (countdownRef.current) {
      clearInterval(countdownRef.current);
    }

    countdownRef.current = setInterval(() => {
      const now = new Date();
      const deathDate = new Date(deathData.death_date);
      const totalSeconds = Math.floor((deathDate - now) / 1000);
      
      if (totalSeconds <= 0) {
        clearInterval(countdownRef.current);
        setCountdown({
          days: 0,
          hours: 0,
          minutes: 0,
          seconds: 0
        });
        return;
      }
      
      const days = Math.floor(totalSeconds / (3600 * 24));
      const hours = Math.floor((totalSeconds % (3600 * 24)) / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = Math.floor(totalSeconds % 60);
      
      setCountdown({
        days,
        hours,
        minutes,
        seconds
      });
    }, 1000);
  };

  const handleHoldStart = () => {
    const timer = setTimeout(() => {
      setHoldAnimation(true);
    }, 5000);
    
    setHoldTimer(timer);
  };

  const handleHoldEnd = () => {
    if (holdTimer) {
      clearTimeout(holdTimer);
      setHoldTimer(null);
    }
    setHoldAnimation(false);
  };

  const calculateAge = () => {
    if (!deathData) return null;
    
    const birthDate = new Date(
      parseInt(formData.dob_year),
      parseInt(formData.dob_month) - 1,
      parseInt(formData.dob_day)
    );
    
    const today = new Date();
    let years = today.getFullYear() - birthDate.getFullYear();
    let months = today.getMonth() - birthDate.getMonth();
    let days = today.getDate() - birthDate.getDate();
    
    if (days < 0) {
      months--;
      days += new Date(today.getFullYear(), today.getMonth(), 0).getDate();
    }
    
    if (months < 0) {
      years--;
      months += 12;
    }
    
    return {
      years,
      months,
      days,
      totalDays: Math.floor((today - birthDate) / (1000 * 60 * 60 * 24)),
      totalWeeks: Math.floor((today - birthDate) / (1000 * 60 * 60 * 24 * 7)),
      totalMonths: years * 12 + months
    };
  };

  const calculateLifespan = () => {
    if (!deathData) return null;
    
    const deathDate = new Date(deathData.death_date);
    const birthDate = new Date(
      parseInt(formData.dob_year),
      parseInt(formData.dob_month) - 1,
      parseInt(formData.dob_day)
    );
    
    let years = deathDate.getFullYear() - birthDate.getFullYear();
    let months = deathDate.getMonth() - birthDate.getMonth();
    let days = deathDate.getDate() - birthDate.getDate();
    
    if (days < 0) {
      months--;
      days += new Date(deathDate.getFullYear(), deathDate.getMonth(), 0).getDate();
    }
    
    if (months < 0) {
      years--;
      months += 12;
    }
    
    return {
      years,
      months,
      days
    };
  };

  const age = calculateAge();
  const lifespan = calculateLifespan();
  
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString('en-US', options);
  };

  if (!isOpen) return null;

  return (
    <div className="clock-overlay">
      <div className={`clock-container ${holdAnimation ? 'pulse-animation' : ''}`}>
        <button className="close-btn" onClick={onClose}>×</button>
        
        <h1 className="clock-title">THE DEATH CLOCK</h1>
        
        {isLoading ? (
          <div className="loading">Calculating your fate...</div>
        ) : debugInfo ? (
          <div className="error-debug">
            <h3>Debug Information</h3>
            <pre>{JSON.stringify(debugInfo, null, 2)}</pre>
            <button onClick={() => setDebugInfo(null)} className="retake-btn">Retry</button>
          </div>
        ) : deathData ? (
          <div className="death-results">
            <div className="test-date">
              Test taken: {new Date().toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}.
            </div>
            
            {age && (
              <div className="current-age">
                <p>At time of testing you are {age.years} years, {age.months} months and {age.days} days old.</p>
                <p className="age-details">Current age in: Days: ({age.totalDays}), Weeks: ({age.totalWeeks}), Months: ({age.totalMonths})</p>
              </div>
            )}
            
            <div className="death-prediction">
              <p>Based on our calculations you will die on: <span className="death-date">{formatDate(deathData.death_date)}</span></p>
            </div>
            
            
            
            {lifespan && (
              <div className="life-expectancy">
                <h2>You will live to be {lifespan.years} years, {lifespan.months} months and {lifespan.days} days old!</h2>
              </div>
            )}
            
            <div className="countdown">
              <p className="countdown-text">
                That's <span className="countdown-number">{countdown.days}</span> Days, <span className="countdown-number">{countdown.hours}</span> Hours, <span className="countdown-number">{countdown.minutes}</span> Minutes, <span className="countdown-number">{countdown.seconds}</span> Seconds remaining...
              </p>
              <p>Or approx: <span className="years-left">{Math.floor(countdown.days / 365)}</span> years</p>
            </div>
            
            <div className="comparisons">
              <p>Avg life expectancy of other {formData.sex} testers from {formData.country || "your region"} with your BMI: {(lifespan?.years - 5 + Math.random() * 10).toFixed(1)} years old</p>
            </div>
            
            <div className="tombstone">
              <div className="tombstone-image">
                <div className="epitaph">
                  <p>in loving memory</p>
                  <p>Taken from us on</p>
                  <p>{formatDate(deathData.death_date)}</p>
                  <p>rest in peace</p>
                </div>
              </div>
            </div>
            
            <div className="actions">
              <button 
                className="reaction-btn"
                onMouseDown={handleHoldStart}
                onMouseUp={handleHoldEnd}
                onMouseLeave={handleHoldEnd}
                onTouchStart={handleHoldStart}
                onTouchEnd={handleHoldEnd}
              >
                Send us your reaction
              </button>
              <span className="separator">|</span>
              <button className="retake-btn" onClick={() => setDeathData(null)}>Retake your test</button>
            </div>
          </div>
        ) : (
          <form className="clock-form" onSubmit={calculateDeathDate}>
            <div className="form-group">
              <label>Date of Birth* (m/d/y)</label>
              <div className="dob-inputs">
                <select name="dob_month" value={formData.dob_month} onChange={handleInputChange}>
                  {Array.from({ length: 12 }, (_, i) => i + 1).map(month => (
                    <option key={month} value={month}>{month}</option>
                  ))}
                </select>
                <select name="dob_day" value={formData.dob_day} onChange={handleInputChange}>
                  {Array.from({ length: 31 }, (_, i) => i + 1).map(day => (
                    <option key={day} value={day}>{day}</option>
                  ))}
                </select>
                <select name="dob_year" value={formData.dob_year} onChange={handleInputChange}>
                  {Array.from({ length: 100 }, (_, i) => new Date().getFullYear() - i).map(year => (
                    <option key={year} value={year}>{year}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-group">
              <label>Sex*</label>
              <select name="sex" value={formData.sex} onChange={handleInputChange}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group checkbox">
              <input 
                type="checkbox" 
                id="smoker" 
                name="smoker" 
                checked={formData.smoker} 
                onChange={handleInputChange} 
              />
              <label htmlFor="smoker">Do you smoke?</label>
            </div>

            <div className="form-group">
              <label>BMI*</label>
              <select name="bmi" value={formData.bmi} onChange={handleInputChange}>
                <option value="<25">&lt;25</option>
                <option value="25-30">25-30</option>
                <option value=">30">&gt;30</option>
              </select>
            </div>

            <div className="form-group">
              <label>Outlook</label>
              <select name="outlook" value={formData.outlook} onChange={handleInputChange}>
                <option value="">Choose...</option>
                <option value="Positive">Positive</option>
                <option value="Neutral">Neutral</option>
                <option value="Negative">Negative</option>
              </select>
            </div>

            <div className="form-group">
              <label>Alcohol Consumption</label>
              <select name="alcohol" value={formData.alcohol} onChange={handleInputChange}>
                <option value="">Choose...</option>
                <option value="None">None</option>
                <option value="Moderate">Moderate</option>
                <option value="Heavy">Heavy</option>
              </select>
            </div>

            <div className="form-group">
              <label>Country*</label>
              <select name="country" value={formData.country} onChange={handleInputChange} required>
                <option value="">Choose...</option>
                <option value="United States">United States</option>
                <option value="United Kingdom">United Kingdom</option>
                <option value="Canada">Canada</option>
                <option value="Australia">Australia</option>
                <option value="Japan">Japan</option>
                <option value="Germany">Germany</option>
                <option value="France">France</option>
                <option value="Italy">Italy</option>
                <option value="Spain">Spain</option>
                <option value="India">India</option>
                <option value="China">China</option>
                <option value="Brazil">Brazil</option>
                <option value="Russia">Russia</option>
                <option value="South Africa">South Africa</option>
              </select>
            </div>

            <div className="form-group checkbox">
              <input 
                type="checkbox" 
                id="include_fitness_diet" 
                name="include_fitness_diet" 
                checked={formData.include_fitness_diet} 
                onChange={handleInputChange} 
              />
              <label htmlFor="include_fitness_diet">Include fitness & diet?</label>
            </div>

            {formData.include_fitness_diet && (
              <>
                <div className="form-group diet-fitness">
                  <label>Level of Fitness</label>
                  <select name="fitness_level" value={formData.fitness_level} onChange={handleInputChange}>
                    <option value="">Choose...</option>
                    <option value="Poor">Poor</option>
                    <option value="Average">Average</option>
                    <option value="Good">Good</option>
                    <option value="Excellent">Excellent</option>
                  </select>
                </div>

                <div className="form-group diet-fitness">
                  <label>Diet Rating</label>
                  <select name="diet_rating" value={formData.diet_rating} onChange={handleInputChange}>
                    <option value="">Choose...</option>
                    <option value="Poor">Poor</option>
                    <option value="Average">Average</option>
                    <option value="Good">Good</option>
                    <option value="Excellent">Excellent</option>
                  </select>
                </div>
              </>
            )}

            <button type="submit" className="submit-btn">Submit</button>
          </form>
        )}
      </div>
    </div>
  );
};

export default TheClockComponent;