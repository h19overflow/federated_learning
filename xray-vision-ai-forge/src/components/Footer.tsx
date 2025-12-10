
import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-white border-t border-gray-200 py-4 mt-8">
      <div className="container">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} Pneumonia Detection System - Medical AI Research
          </p>
          <div className="flex space-x-4 mt-2 md:mt-0">
            <Link to="/" className="text-sm text-medical hover:underline">Home</Link>
            <Link to="/saved-experiments" className="text-sm text-medical hover:underline">Saved Experiments</Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
