import React, { useState, useEffect, useContext } from "react";
import { motion } from "framer-motion";
import { X, Clock, Check, Calendar } from "lucide-react";
import { AuthContext } from "../context/AuthContext";

const BookAppointment = ({ doctor, onClose, onAppointmentBooked }) => {
  const { user } = useContext(AuthContext);
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [selectedDay, setSelectedDay] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [bookingStatus, setBookingStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableSlots, setAvailableSlots] = useState({});
  const [availableDates, setAvailableDates] = useState([]);

  useEffect(() => {
    fetchAvailableSlots();
    generateAvailableDates();
  }, [doctor]);

  // Generate dates starting from today up to 3 days in the future
  const generateAvailableDates = () => {
    const dates = [];
    const today = new Date();
    const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    
    // Include today and 3 days after (total 4 days)
    for (let i = 0; i <= 3; i++) {
      const date = new Date();
      date.setDate(today.getDate() + i);
      const dayName = dayNames[date.getDay()];
      dates.push({
        date: date,
        formattedDate: date.toLocaleDateString('en-US', { 
          month: 'short', 
          day: 'numeric',
          weekday: 'short'
        }),
        dayName
      });
    }
    setAvailableDates(dates);
  };

  const fetchAvailableSlots = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/doctor/${doctor?._id}/slots`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
      });
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      setAvailableSlots(data.available_slots || {});
    } catch (error) {
      console.error("Error fetching slots:", error);
    }
  };

  const bookAppointment = async () => {
    if (!selectedSlot || !selectedDay || !selectedDate) return;
    setLoading(true);
  
    // Check if user ID is available
    if (!user || !user.id) {
      console.error("DEBUG - User ID is missing:", user);
      setBookingStatus("error");
      setLoading(false);
      return;
    }
  
    try {
      // Format the date string for the API
      const formattedDate = selectedDate.date.toISOString().split('T')[0];
      
      console.log("DEBUG - User context:", user);
      console.log("DEBUG - Sending data:", {
        user_id: user.id,
        doctor_id: doctor?._id,
        day: `${selectedDay} (${formattedDate})`,
        slot: selectedSlot,
      });
  
      const response = await fetch("http://127.0.0.1:5000/user/book-appointment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({
          user_id: user.id,
          doctor_id: doctor?._id,
          day: `${selectedDay} (${formattedDate})`,
          slot: selectedSlot,
        }),
      });
  
      // Check if response is JSON
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        console.error("DEBUG - Non-JSON response received:", await response.text());
        setBookingStatus("error");
        setLoading(false);
        return;
      }
  
      const data = await response.json();
      console.log("DEBUG - Server response:", data);
      
      if (response.ok) {
        setBookingStatus("success");
        onAppointmentBooked(data.appointment);
        fetchAvailableSlots();
      } else {
        setBookingStatus("error");
        console.error("Booking error:", data);
      }
    } catch (error) {
      console.error("Booking error:", error);
      setBookingStatus("error");
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <motion.div 
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div 
        className="bg-white rounded-xl w-full max-w-md overflow-hidden"
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 50, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-bold text-gray-800">Book Appointment</h2>
          <button className="bg-gray-100 p-2 rounded-full hover:bg-gray-200" onClick={onClose}>
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4">
          {/* Date Selection */}
          <div className="mb-4">
            <p className="text-sm text-gray-700 mb-2">Select a date:</p>
            <div className="flex overflow-x-auto gap-2 pb-2">
              {availableDates.map((dateObj, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setSelectedDate(dateObj);
                    setSelectedDay(dateObj.dayName);
                    setSelectedSlot(null);
                  }}
                  className={`px-3 py-2 rounded text-xs border flex-shrink-0 flex flex-col items-center 
                    ${selectedDate === dateObj
                      ? "bg-blue-500 text-white border-blue-500" 
                      : "bg-gray-100 text-gray-800 border-gray-300"}`}
                >
                  <Calendar className="w-3 h-3 mb-1" />
                  <span>{dateObj.formattedDate}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Slot Selection */}
          <p className="text-sm text-gray-700 mb-3">Select an available time slot:</p>

          {selectedDay && Object.keys(availableSlots).length > 0 && availableSlots[selectedDay] ? (
            <div className="mb-3">
              <p className="text-sm font-semibold text-gray-800">{selectedDay}</p>
              <div className="flex flex-wrap gap-2">
                {availableSlots[selectedDay].length > 0 ? (
                  availableSlots[selectedDay].map((slot, index) => (
                    <button
                      key={index}
                      onClick={() => setSelectedSlot(slot)}
                      className={`px-3 py-1 rounded text-xs border transition-colors 
                        ${selectedSlot === slot
                          ? "bg-blue-500 text-white border-blue-500" 
                          : "bg-gray-100 text-gray-800 border-gray-300"}`}
                    >
                      <Clock className="w-3 h-3 inline mr-1" /> {slot}
                    </button>
                  ))
                ) : (
                  <p className="text-gray-500 text-xs">No available slots</p>
                )}
              </div>
            </div>
          ) : selectedDay ? (
            <p className="text-gray-500 text-xs">No available slots for {selectedDay}</p>
          ) : (
            <p className="text-gray-500 text-xs">Please select a date first</p>
          )}

          {bookingStatus === "error" && (
            <p className="mt-3 text-xs text-red-500">Failed to book appointment. Try again.</p>
          )}

          {bookingStatus === "success" && (
            <p className="mt-3 text-xs text-green-500 flex items-center gap-1">
              <Check className="w-3 h-3" /> Appointment booked successfully!
            </p>
          )}

          <div className="mt-4 flex justify-end">
            <button
              onClick={bookAppointment}
              disabled={!selectedSlot || !selectedDate || loading}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors text-sm disabled:opacity-50"
            >
              {loading ? "Booking..." : "Confirm Appointment"}
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default BookAppointment;