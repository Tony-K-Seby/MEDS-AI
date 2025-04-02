import React, { useEffect, useState } from "react";
import Navbar from "./Navbar";
import doctorImage from "../assets/doctor.jpg"; // Default doctor image
import Squares from "./Squares";
import { motion, AnimatePresence } from "framer-motion"; // For animations
import { Star, MapPin, Calendar, Heart, X, Clock, Award, Bookmark, MessageSquare, Phone } from "lucide-react"; // For icons
import BookAppointment from "./BookAppointment"; // Import the BookAppointment component

const API_URL = "http://127.0.0.1:5000/user/doctors"; // Corrected API URL

const DoctorDetailModal = ({ doctor, onClose }) => {
  const [showBookAppointment, setShowBookAppointment] = useState(false);

  const openAppointmentModal = () => {
    setShowBookAppointment(true);
  };

  const closeAppointmentModal = () => {
    setShowBookAppointment(false);
  };

  return (
    <>
      <motion.div 
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 overflow-y-auto"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div 
          className="bg-white rounded-xl w-full max-w-2xl overflow-hidden my-8"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 50, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-start p-4">
            <div className="w-24 h-24 rounded-lg overflow-hidden flex-shrink-0">
              <img
                src={doctor.image || doctorImage} 
                alt={doctor.name}
                className="w-full h-full object-cover"
              />
            </div>
            
            <div className="ml-4 flex-1">
              <h2 className="text-xl font-bold text-gray-800">{doctor.name}</h2>
              
              <div className="flex items-center gap-2 mt-1">
                <Award className="w-4 h-4 text-blue-500 flex-shrink-0" />
                <span className="text-gray-700 font-medium text-sm">{doctor.specialization_name}</span>
              </div>
              
              <div className="flex items-center gap-2 mt-1">
                <MapPin className="w-4 h-4 text-green-500 flex-shrink-0" />
                <span className="text-gray-700 text-sm">{doctor.hospital_name}</span>
              </div>
              
              <div className="flex items-center gap-2 mt-1">
                <Phone className="w-4 h-4 text-blue-500 flex-shrink-0" />
                <span className="text-gray-700 text-sm">{doctor.contact_no}</span>
              </div>
            </div>
            
            <button 
              className="bg-gray-100 p-2 rounded-full hover:bg-gray-200 transition-colors flex-shrink-0"
              onClick={onClose}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          
          <div className="px-4 pb-4">          
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
              <div>
                <h3 className="font-semibold text-gray-800 text-sm">Education & Experience</h3>
                <p className="text-gray-600 text-xs mt-1">{doctor.education}</p>
                <p className="text-gray-600 text-xs">{doctor.experience} of experience</p>
                
                <h3 className="font-semibold text-gray-800 text-sm mt-3">Languages</h3>
                <div className="flex flex-wrap gap-1 mt-1">
                  {doctor.languages && doctor.languages.map((lang, index) => (
                    <span key={index} className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded text-xs">
                      {lang}
                    </span>
                  ))}
                </div>
                
                <h3 className="font-semibold text-gray-800 text-sm mt-3">Hospital Address</h3>
                <p className="text-gray-600 text-xs mt-1">{doctor.address}</p>
              </div>
              
              <div>
                <h3 className="font-semibold text-gray-800 text-sm">Procedures</h3>
                <div className="flex flex-wrap gap-1 mt-1">
                  {doctor.procedures && doctor.procedures.map((proc, index) => (
                    <span key={index} className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-xs">
                      {proc}
                    </span>
                  ))}
                </div>
                
                <h3 className="font-semibold text-gray-800 text-sm mt-3">Availability</h3>
                <div className="mt-1">
                  {doctor.availability && doctor.availability.map((slot, index) => (
                    <div key={index} className="flex items-center gap-2 mb-1">
                      <Clock className="w-3 h-3 text-green-500 flex-shrink-0" />
                      <span className="text-gray-600 text-xs">{slot}</span>
                    </div>
                  ))}
                </div>
                
                <h3 className="font-semibold text-gray-800 text-sm mt-3">Awards</h3>
                <div className="mt-1">
                  {doctor.awards && doctor.awards.map((award, index) => (
                    <div key={index} className="flex items-center gap-2 mb-1">
                      <Award className="w-3 h-3 text-yellow-500 flex-shrink-0" />
                      <span className="text-gray-600 text-xs">{award}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex gap-2 mt-4">
              <button 
                className="flex-1 bg-blue-500 hover:bg-blue-600 text-white px-3 py-2 rounded-lg transition-colors flex items-center justify-center gap-1 text-sm"
                onClick={openAppointmentModal}
              >
                <Calendar className="w-4 h-4" />
                <span>Book Appointment</span>
              </button>
              <button className="bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-2 rounded-lg transition-colors flex items-center justify-center gap-1 text-sm">
                <MessageSquare className="w-4 h-4" />
                <span>Message</span>
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>

      <AnimatePresence>
        {showBookAppointment && (
          <BookAppointment 
            doctor={doctor} 
            onClose={closeAppointmentModal} 
            onAppointmentBooked={(appointment) => {
              // Optionally, you can do something with the booked appointment
              closeAppointmentModal();
            }}
          />
        )}
      </AnimatePresence>
    </>
  );
};

const DoctorCard = ({ doctor, onClick }) => {
  const [favorite, setFavorite] = useState(false);
  
  const toggleFavorite = (e) => {
    e.stopPropagation();
    setFavorite(!favorite);
  };

  return (
    <motion.div 
      className="backdrop-blur-lg bg-white/10 shadow-md rounded-xl border border-white/30 relative overflow-hidden cursor-pointer w-full"
      whileHover={{ scale: 1.01 }}
      transition={{ duration: 0.2 }}
      onClick={() => onClick(doctor)}
    >
      <div className="flex p-3">
        {/* Square image */}
        <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0">
          <img
            src={doctor.image || doctorImage} 
            alt={doctor.name}
            className="w-full h-full object-cover"
          />
        </div>
        
        <div className="ml-3 flex-1 min-w-0">
          <h2 className="text-base font-semibold text-[#242120] truncate">{doctor.name}</h2>
          <div className="flex items-center gap-1 mt-0.5">
            <Award className="w-3 h-3 text-blue-400 flex-shrink-0" />
            <p className="text-[#242120] font-medium text-xs truncate">{doctor.specialization_name}</p>
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            <MapPin className="w-3 h-3 text-green-400 flex-shrink-0" />
            <p className="text-[#242120] text-xs truncate">{doctor.hospital_name}</p>
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            <span className="w-3 h-3 text-blue-300 flex-shrink-0">📞</span>
            <span className="text-[#050505] font-medium text-xs truncate">{doctor.contact_no}</span>
          </div>
        </div>
        
        <button 
          className={`p-1.5 h-min rounded-full bg-white/70 hover:bg-white transition-colors flex-shrink-0 ${favorite ? 'text-red-500' : 'text-gray-500'}`}
          onClick={toggleFavorite}
        >
          <Heart className={`w-3.5 h-3.5 ${favorite ? 'fill-current' : ''}`} />
        </button>
      </div>
      
      <button className="w-full bg-blue-500 hover:bg-blue-600 text-white px-3 py-1.5 transition-colors flex items-center justify-center gap-1 text-xs rounded-t-none rounded-b-xl">
        <Calendar className="w-3 h-3" />
        <span>View Profile</span>
      </button>
    </motion.div>
  );
};

const DoctorList = () => {
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [view, setView] = useState('grid'); // 'grid' or 'map'

  useEffect(() => {
    fetch(API_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch doctors");
        }
        return response.json();
      })
      .then((data) => {
        setDoctors(data.doctors || []);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching doctors:", error);
        setError(error.message);
        setLoading(false);
      });
  }, []);

  const openDoctorDetails = (doctor) => {
    setSelectedDoctor(doctor);
    // Prevent scrolling when modal is open
    document.body.style.overflow = "hidden";
  };

  const closeDoctorDetails = () => {
    setSelectedDoctor(null);
    // Re-enable scrolling
    document.body.style.overflow = "auto";
  };

  return (
    <div className="relative min-h-screen max-h-screen flex flex-col bg-transparent px-1 py-1 overflow-hidden">
      <div className="absolute inset-0 w-full h-full z-0 pointer-events-none">
        <Squares
          speed={0.05}
          squareSize={20}
          direction="diagonal"
          borderColor="#ffc05c"
          hoverFillColor="#222"
        />
      </div>

      <div className="flex flex-col items-center w-full z-10 overflow-hidden">
        {/* View toggle buttons */}
        <div className="flex justify-center gap-2 my-2 w-full max-w-xs mx-auto">
          <button 
            className={`flex-1 px-3 py-1.5 rounded-lg transition-colors text-sm ${view === 'grid' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setView('grid')}
          >
            Grid View
          </button>
          <button 
            className={`flex-1 px-3 py-1.5 rounded-lg transition-colors text-sm ${view === 'map' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setView('map')}
          >
            Map View
          </button>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center h-64">
            <div className="w-10 h-10 border-3 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-center text-sm font-semibold mt-3">Loading doctors...</p>
          </div>
        ) : error ? (
          <p className="text-center text-red-500 mt-4 bg-red-50 px-3 py-2 rounded-lg border border-red-200 mx-2 text-sm">
            {error}
          </p>
        ) : view === 'grid' ? (
          <motion.div 
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 p-2 w-full max-w-7xl mx-auto overflow-y-auto max-h-[calc(100vh-80px)]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {doctors.map((doctor, index) => (
              <DoctorCard 
                key={doctor._id || index} 
                doctor={doctor} 
                onClick={openDoctorDetails} 
              />
            ))}
          </motion.div>
        ) : (
          <div className="w-full max-w-7xl mx-auto p-2">
            <div className="bg-white rounded-xl shadow-lg h-64 flex items-center justify-center">
              <p className="text-gray-500 text-sm">Map view coming soon</p>
            </div>
          </div>
        )}
      </div>

      {/* Doctor Detail Modal */}
      <AnimatePresence>
        {selectedDoctor && (
          <DoctorDetailModal 
            doctor={selectedDoctor} 
            onClose={closeDoctorDetails} 
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default DoctorList;
