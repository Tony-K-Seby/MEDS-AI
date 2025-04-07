


import React, { useState, useEffect, useContext } from "react";
import { AuthContext } from "../context/AuthContext"; // Make sure you have this import
import SymptomPredictor from "./SymptomPredictor";
import FloatingChatBot from "./FloatingChatBot";
import Squares from "./Squares";
import BookAppointment from "./BookAppointment"; // Import the BookAppointment component

const Predict = () => {
  const { user, isAuthenticated } = useContext(AuthContext); // Get auth context
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [precautions, setPrecautions] = useState([]);
  const [precautionsLoading, setPrecautionsLoading] = useState(false);
  // States for appointment booking
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [showBookingModal, setShowBookingModal] = useState(false);

  const handlePredictionStart = () => {
    setIsPredicting(true);
    setPrediction(null);
    setDoctors([]);
    setPrecautions([]);
  };

  const fetchPrecautions = async (diseases) => {
    setPrecautionsLoading(true);
    try {
      if (!Array.isArray(diseases)) {
        diseases = [diseases]; // Ensure diseases is always an array
      }
  
      const precautionsData = await Promise.all(
        diseases.map(async (disease) => {
          const response = await fetch(`http://127.0.0.1:5000/user/disease/precautions?disease=${encodeURIComponent(disease)}`);
          if (!response.ok) {
            throw new Error(`Failed to fetch precautions for ${disease}`);
          }
          return response.json();
        })
      );
  
      console.log("Precautions:", precautionsData);
      setPrecautions(precautionsData);
    } catch (error) {
      console.error("Error fetching precautions:", error);
      setPrecautions([]);
    } finally {
      setPrecautionsLoading(false);
    }
  };
  
  const handlePredictionResult = async (predictionData) => {
    setIsPredicting(false);
    
    // Ensure predictionData is an object, take the first element if it's an array
    const finalPrediction = Array.isArray(predictionData) ? predictionData[0] : predictionData;
    setPrediction(finalPrediction);
    
    // Fetch doctors only if user is authenticated
    if (isAuthenticated && finalPrediction && finalPrediction.specialty_id) {
      fetchDoctors(finalPrediction.specialty_id);
    }
    
    if (finalPrediction && finalPrediction.final_prediction) {
      const diseases = Array.isArray(finalPrediction.final_prediction)
        ? finalPrediction.final_prediction
        : [finalPrediction.final_prediction]; // Convert to array if it's a string
      
      fetchPrecautions(diseases);
    }
  };

  const fetchDoctors = async (specialtyId) => {
    // Only fetch doctors if user is authenticated
    if (!isAuthenticated) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `http://127.0.0.1:5000/user/doctors/specialty/${specialtyId}`,
        {
          headers: {
            "Authorization": `Bearer ${localStorage.getItem("token")}`,
          }
        }
      );
      if (!response.ok) {
        throw new Error("Failed to fetch doctors");
      }
      const data = await response.json();
      setDoctors(data.doctors || []);
    } catch (err) {
      console.error("Error fetching doctors:", err);
      setError("Failed to load doctors. Please try again.");
      setDoctors([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle booking appointment
  const handleBookAppointment = (doctor) => {
    setSelectedDoctor(doctor);
    setShowBookingModal(true);
  };

  const handleCloseBookingModal = () => {
    setShowBookingModal(false);
    setSelectedDoctor(null);
  };

  const handleAppointmentBooked = (appointment) => {
    // You could update UI or show a confirmation message
    console.log("Appointment booked:", appointment);
    // Close modal after a short delay
    setTimeout(() => {
      setShowBookingModal(false);
      setSelectedDoctor(null);
    }, 2000);
  };

  // Glassmorphism style
  const glassmorphismStyle = {
    background: "rgba(255, 255, 255, 0.25)",
    backdropFilter: "blur(10px)",
    WebkitBackdropFilter: "blur(10px)",
    borderRadius: "10px",
    border: "1px solid rgba(255, 255, 255, 0.18)",
    boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.37)"
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-transparent">
      {/* Background Animation */}
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
          direction="diagonal"
          borderColor="#ffc05c"
          hoverFillColor="#222"
        />
      </div>

      {/* Two-column layout with independent heights */}
      <div className="container mx-auto px-4 py-8 z-10">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 auto-rows-auto">
          {/* Prediction Column - Height based on content */}
          <div className="prediction-column">
            <div style={glassmorphismStyle} className="p-6">
              <h2 className="text-2xl font-bold mb-4 text-white">Symptom Analysis</h2>
              <SymptomPredictor 
                onPredictionStart={handlePredictionStart}
                onPredictionResult={handlePredictionResult} 
              />
            </div>
          </div>

          {/* Right Column */}
          <div className="right-column space-y-6">
            {/* Precautions Box */}
            <div style={glassmorphismStyle} className="p-6">
              <h2 className="text-2xl font-bold mb-4 text-black">Disease Precautions</h2>
              
              {isPredicting ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="animate-pulse mb-4">
                    <svg className="w-16 h-16 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                  </div>
                  <p className="text-lg text-black">Processing your symptoms...</p>
                </div>
              ) : precautionsLoading ? (
                <div className="flex justify-center items-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                </div>
              ) : precautions && precautions.length > 0 ? (
                <div className="space-y-6">
                  {precautions.map((item, idx) => (
                    <div key={idx} className="space-y-4">
                      <h3 className="text-xl font-semibold text-black mb-2">
                        {item.disease} Precautions
                      </h3>
                      <ul className="list-disc list-inside text-black space-y-2">
                        {item.precautions
                          .filter(precaution => precaution !== "nan")
                          .map((precaution, index) => (
                            <li key={index} className="text-sm">
                              {precaution}
                            </li>
                          ))
                        }
                      </ul>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <svg className="w-16 h-16 text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                  </svg>
                  <p className="text-lg text-white">No precautions found for this condition</p>
                </div>
              )}
            </div>

            {/* Recommended Doctors Box - Only show when logged in */}
            {isAuthenticated && (
              <div style={glassmorphismStyle} className="p-6">
                <h2 className="text-2xl font-bold mb-4 text-white">Recommended Doctors</h2>
                
                {isPredicting ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="animate-pulse mb-4">
                      <svg className="w-16 h-16 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                      </svg>
                    </div>
                    <p className="text-lg text-white">Processing your symptoms...</p>
                    <p className="text-sm text-gray-200 mt-2">We'll find the right specialists for you.</p>
                  </div>
                ) : loading ? (
                  <div className="flex justify-center items-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                  </div>
                ) : error ? (
                  <div className="bg-red-100 bg-opacity-50 backdrop-blur-sm border border-red-400 text-red-700 px-4 py-3 rounded">
                    <p>{error}</p>
                  </div>
                ) : doctors.length > 0 ? (
                  <div className="space-y-4">
                    {doctors.map((doctor) => (
                      <div
                        key={doctor._id}
                        className="bg-blue-600 bg-opacity-25 backdrop-blur-sm border-l-4 border-l-blue-400 rounded-lg p-4 hover:shadow-lg transition-all duration-300"
                      >
                        <h3 className="text-xl font-semibold text-white">{doctor.name}</h3>
                        <p className="text-blue-100 mb-2 font-medium">{doctor.qualification}</p>
                        <p className="text-sm text-gray-200">{doctor.address}</p>
                        <div className="mt-3 flex justify-between items-center">
                          <span className="text-blue-200 font-medium">{doctor.contact_no}</span>
                          <button 
                            onClick={() => handleBookAppointment(doctor)}
                            className="bg-gradient-to-r from-[#ff8c42] to-[#ff3e55] text-white px-4 py-2 rounded hover:from-[#ff3e55] hover:to-[#ff8c42] transition-all duration-300"
                          >
                            Book Appointment
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : prediction ? (
                  <div className="text-center py-8">
                    <p className="text-gray-200">No doctors found for this condition.</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <svg className="w-16 h-16 text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                    </svg>
                    <p className="text-lg text-white">Enter your symptoms</p>
                    <p className="text-sm text-gray-200 mt-2">Doctor recommendations will appear here</p>
                  </div>
                )}
              </div>
            )}
            
            {/* Not logged in message - Show when not authenticated */}
            {!isAuthenticated && prediction && (
              <div style={glassmorphismStyle} className="p-6">
                <div className="text-center py-6">
                  <svg className="w-12 h-12 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                  </svg>
                  <h3 className="text-xl font-semibold text-white mb-2">Login to See Doctor Recommendations</h3>
                  <p className="text-gray-200">Sign in to see specialists for your condition and book appointments.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Floating ChatBot */}
      <FloatingChatBot />

      {/* Booking Appointment Modal */}
      {showBookingModal && selectedDoctor && (
        <BookAppointment 
          doctor={selectedDoctor}
          onClose={handleCloseBookingModal}
          onAppointmentBooked={handleAppointmentBooked}
        />
      )}
    </div>
  );
};

export default Predict;