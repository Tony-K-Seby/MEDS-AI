import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import axios from 'axios';
import { Loader2 } from 'lucide-react';
import Squares from './Squares'; // Added import for Squares component

const Appointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [doctorDetails, setDoctorDetails] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useContext(AuthContext);
  
  // State for modal popup
  const [showModal, setShowModal] = useState(false);
  const [appointmentToCancel, setAppointmentToCancel] = useState(null);

  useEffect(() => {
    const fetchAppointments = async () => {
      try {
        // Fetch user's appointments
        const response = await axios.get(`http://127.0.0.1:5000/user/appointments/${user.id}`);
        console.log('DEBUG - Appointments data:', response.data);
        setAppointments(response.data.appointments);
        
        // Fetch doctors details
        const doctorIds = [...new Set(response.data.appointments.map(app => app.doctor_id))];
        const doctorsData = {};
        
        for (const docId of doctorIds) {
          try {
            // Use your existing detailed doctor endpoint
            const docResponse = await axios.get(`http://127.0.0.1:5000/user/doctors/${docId}`);
            doctorsData[docId] = docResponse.data.doctor;
          } catch (err) {
            console.log(`Could not fetch doctor with ID ${docId}:`, err);
          }
        }
        
        setDoctorDetails(doctorsData);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching appointments:', err);
        setError('Failed to load appointments');
        setLoading(false);
      }
    };

    if (user?.id) {
      fetchAppointments();
    }
  }, [user]);

  const confirmCancelAppointment = async () => {
    if (!appointmentToCancel) return;
    try {
      setLoading(true);
      console.log(`DEBUG - Canceling appointment with ID: ${appointmentToCancel}`);
      await axios.delete(`http://127.0.0.1:5000/user/cancel-appointment/${appointmentToCancel}`);
      
      // Remove the canceled appointment from state
      setAppointments(appointments.filter(app => app._id !== appointmentToCancel));
      
      // Close modal
      setShowModal(false);
      setAppointmentToCancel(null);
      
      setLoading(false);
    } catch (err) {
      console.error('Error canceling appointment:', err);
      // Optionally handle error state
      setShowModal(false);
      setAppointmentToCancel(null);
      setLoading(false);
    }
  };

  const openCancelModal = (appointmentId) => {
    setAppointmentToCancel(appointmentId);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setAppointmentToCancel(null);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <p className="mt-4 text-gray-500">Loading appointments...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex justify-center p-8">
        <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700">
          {error}
        </div>
      </div>
    );
  }
  
  if (appointments.length === 0) {
    return (
      <div className="flex justify-center p-8">
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4 text-blue-700">
          You have no appointments scheduled. Book an appointment to get started.
        </div>
      </div>
    );
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div style={{ position: "relative", overflow: "hidden", minHeight: "100vh" }}>
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
          direction="diagonal"
          borderColor="#ffc05c"
          hoverFillColor="#222"
        />
      </div>
      
      <div className="container mx-auto p-6 md:p-8">
        <h1 className="text-3xl font-bold mb-8 text-center text-gray-800 bg-white bg-opacity-70 p-4 rounded-lg shadow-md">
          My Appointments
        </h1>
        
        <div className="grid gap-6 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-4 justify-items-center">
          {appointments.map((appointment) => {
            const doctor = doctorDetails[appointment.doctor_id] || {};
            
            return (
              <div 
                key={appointment._id} 
                className="rounded-lg overflow-hidden shadow-lg"
                style={{
                  background: "rgba(191, 219, 254, 0.7)", // Tailwind blue-200 with opacity
                  backdropFilter: "blur(10px)",
                  border: "1px solid rgba(255, 255, 255, 0.3)",
                  boxShadow: "0 8px 32px 0 rgba(31, 38, 135, 0.2)",
                  maxWidth: "280px",
                }}
              >
                {/* Card Header */}
                <div className="p-3 border-b border-blue-300 bg-blue-100 bg-opacity-50">
                  <h3 className="text-lg font-bold text-gray-900">
                    {doctor.name || 'Doctor information unavailable'}
                  </h3>
                  <p className="text-xs text-gray-700">
                    {doctor.specialization_name || 'Specialty unknown'}
                  </p>
                </div>
                
                {/* Card Content */}
                <div className="p-3">
                  <div className="space-y-2">
                    <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                      <span className="font-semibold">Day:</span> {appointment.day}
                    </div>
                    <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                      <span className="font-semibold">Time:</span> {appointment.slot}
                    </div>
                    <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                      <span className="font-semibold">Booked on:</span> {formatDate(appointment.booked_at)}
                    </div>
                    
                    {doctor.contact_no && (
                      <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                        <span className="font-semibold">Contact:</span> {doctor.contact_no}
                      </div>
                    )}
                    
                    {doctor.hospital_name && (
                      <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                        <span className="font-semibold">Hospital:</span> {doctor.hospital_name}
                      </div>
                    )}
                    
                    {doctor.hospital_address && (
                      <div className="bg-white bg-opacity-50 p-1 rounded text-xs">
                        <span className="font-semibold">Address:</span> {doctor.hospital_address}
                        {doctor.hospital_city && `, ${doctor.hospital_city}`}
                        {doctor.hospital_state && `, ${doctor.hospital_state}`}
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Card Footer */}
                <div className="p-3 border-t border-blue-300">
                  <button 
                    className="w-full bg-red-600 hover:bg-red-700 text-white text-sm py-2 px-3 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 transition-colors shadow-md"
                    onClick={() => openCancelModal(appointment._id)}
                  >
                    Cancel Appointment
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modal for cancellation confirmation */}
      {showModal && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white rounded-lg p-6 w-80 shadow-lg">
            <h2 className="text-xl font-bold mb-4 text-gray-800">Confirm Cancellation</h2>
            <p className="text-gray-600 mb-6">Are you sure you want to cancel this appointment?</p>
            <div className="flex justify-end space-x-4">
              <button 
                className="px-4 py-2 bg-gray-300 text-gray-800 rounded hover:bg-gray-400 transition-colors"
                onClick={closeModal}
              >
                No
              </button>
              <button 
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                onClick={confirmCancelAppointment}
              >
                Confirm Cancellation
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Appointments;
