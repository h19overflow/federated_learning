import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-white border-t border-[hsl(210_15%_92%)] py-6">
      <div className="container">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {/* Mini logo */}
            <div className="w-8 h-8 rounded-lg bg-[hsl(172_63%_22%)] flex items-center justify-center">
              <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none">
                <path d="M12 4v16M4 12h16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
            <p className="text-sm text-[hsl(215_15%_50%)]">
              {new Date().getFullYear()} XRay Vision AI. Medical AI Research Platform.
            </p>
          </div>

          <div className="flex items-center gap-6">
            <Link
              to="/"
              className="text-sm text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_28%)] transition-colors"
            >
              Home
            </Link>
            <Link
              to="/experiment"
              className="text-sm text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_28%)] transition-colors"
            >
              Experiment
            </Link>
            <Link
              to="/saved-experiments"
              className="text-sm text-[hsl(215_15%_45%)] hover:text-[hsl(172_63%_28%)] transition-colors"
            >
              Saved
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
