import React, { useState } from "react";
import { Send, Phone, Mail, MapPin, Users } from "lucide-react";

// Squares component implementation
const Squares = ({
  speed = 0.2,
  squareSize = 40,
  direction = "diagonal",
  borderColor = "#0066CC", // Updated border color
  hoverFillColor = "#003366", // Updated hover fill color
}) => {
  // Generate squares for the background
  const squareCount = 50; // Number of squares to generate
  const squares = Array(squareCount)
    .fill()
    .map((_, i) => ({
      id: i,
      size: Math.random() * (squareSize * 0.5) + squareSize * 0.5,
      x: Math.random() * 100,
      y: Math.random() * 100,
      rotation: Math.random() * 360,
      speed: (Math.random() * 0.5 + 0.5) * speed,
    }));

  return (
    <div className="w-full h-full bg-gradient-to-br from-gray-900 to-black relative overflow-hidden">
      {squares.map((square) => (
        <div
          key={square.id}
          className="absolute border-2 transition-all duration-300 hover:bg-opacity-30"
          style={{
            width: `${square.size}px`,
            height: `${square.size}px`,
            left: `${square.x}%`,
            top: `${square.y}%`,
            borderColor: borderColor,
            transform: `rotate(${square.rotation}deg)`,
            animation: `float-${direction} ${
              20 / square.speed
            }s infinite linear`,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = hoverFillColor;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "transparent";
          }}
        />
      ))}

      <style jsx>{`
        @keyframes float-diagonal {
          0% {
            transform: translate(0, 0) rotate(${Math.random() * 360}deg);
          }
          100% {
            transform: translate(
                ${Math.random() > 0.5 ? "" : "-"}100px,
                ${Math.random() > 0.5 ? "" : "-"}100px
              )
              rotate(${Math.random() * 360}deg);
          }
        }
        @keyframes float-up {
          0% {
            transform: translateY(0) rotate(${Math.random() * 360}deg);
          }
          100% {
            transform: translateY(-100px) rotate(${Math.random() * 360}deg);
          }
        }
        @keyframes float-down {
          0% {
            transform: translateY(0) rotate(${Math.random() * 360}deg);
          }
          100% {
            transform: translateY(100px) rotate(${Math.random() * 360}deg);
          }
        }
        @keyframes float-left {
          0% {
            transform: translateX(0) rotate(${Math.random() * 360}deg);
          }
          100% {
            transform: translateX(-100px) rotate(${Math.random() * 360}deg);
          }
        }
        @keyframes float-right {
          0% {
            transform: translateX(0) rotate(${Math.random() * 360}deg);
          }
          100% {
            transform: translateX(100px) rotate(${Math.random() * 360}deg);
          }
        }
      `}</style>
    </div>
  );
};

const ContactUsPage = () => {
  const [formData, setState] = useState({
    name: "",
    email: "",
    message: "",
  });

  return (
    <div className="relative w-full h-screen flex items-center justify-center p-4 overflow-hidden">
      {/* Background Squares Component */}
      <div className="absolute top-0 left-0 w-full h-full z-0">
        <Squares
          speed={0.2}
          squareSize={40}
          direction="diagonal"
          borderColor="#0066CC" // Updated border color
          hoverFillColor="#003366" // Updated hover fill color
        />
      </div>

      {/* Glassmorphism container */}
      <div className="relative z-10 w-full max-w-4xl flex flex-col md:flex-row rounded-2xl overflow-hidden shadow-2xl">
        {/* Contact form section */}
        <div className="w-full md:w-3/5 bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg p-8 border border-white border-opacity-20">
          <h2 className="text-3xl font-bold mb-6 text-white">Contact Us</h2>

          <form className="space-y-6">
            <div>
              <label className="block text-white mb-2">Name</label>
              <input
                type="text"
                className="w-full p-3 bg-white bg-opacity-10 rounded-lg border border-white border-opacity-20 text-white focus:outline-none focus:ring-2 focus:ring-[#0066CC]" // Updated focus ring color
                placeholder="Your name"
              />
            </div>

            <div>
              <label className="block text-white mb-2">Email</label>
              <input
                type="email"
                className="w-full p-3 bg-white bg-opacity-10 rounded-lg border border-white border-opacity-20 text-white focus:outline-none focus:ring-2 focus:ring-[#0066CC]" // Updated focus ring color
                placeholder="your@email.com"
              />
            </div>

            <div>
              <label className="block text-white mb-2">Message</label>
              <textarea
                className="w-full p-3 bg-white bg-opacity-10 rounded-lg border border-white border-opacity-20 text-white h-32 resize-none focus:outline-none focus:ring-2 focus:ring-[#0066CC]" // Updated focus ring color
                placeholder="How can we help you?"
              ></textarea>
            </div>

            <button className="w-full bg-[#0066CC] hover:bg-[#004C99] text-white font-bold py-3 px-4 rounded-lg transition duration-300 flex items-center justify-center">
              <Send className="mr-2 h-5 w-5" />
              Send Message
            </button>
          </form>
        </div>

        {/* Info section */}
        <div className="w-full md:w-2/5 bg-gradient-to-br from-[#0066CC] to-[#004C99] p-8 flex flex-col justify-between">
          <div>
            <h3 className="text-2xl font-bold text-white mb-8">Get in touch</h3>

            <div className="space-y-6">
              <div className="flex items-center">
                <div className="bg-white bg-opacity-20 p-3 rounded-full mr-4">
                  <Mail className="h-5 w-5 text-white" />
                </div>
                <div>
                  <p className="text-white opacity-70">Email</p>
                  <p className="text-white font-semibold">medsai@gmail.com</p>
                </div>
              </div>

              <div className="flex items-center">
                <div className="bg-white bg-opacity-20 p-3 rounded-full mr-4">
                  <MapPin className="h-5 w-5 text-white" />
                </div>
                <div>
                  <p className="text-white opacity-70">Address</p>
                  <p className="text-white font-semibold">
                    Christ College of Engineering
                    <br />
                    Irinjalakuda, Kerala
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContactUsPage;
