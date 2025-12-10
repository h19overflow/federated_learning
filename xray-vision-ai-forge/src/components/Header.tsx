
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Stethoscope, Save, HelpCircle, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from '@/components/ui/navigation-menu';

interface HeaderProps {
  onShowHelp?: () => void;
}

const Header = ({ onShowHelp }: HeaderProps = {}) => {
  const location = useLocation();
  const isExperimentPage = location.pathname === '/experiment';

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm py-4">
      <div className="container flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Link to="/" className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
            <Stethoscope className="h-8 w-8 text-medical-dark" />
            <h1 className="text-2xl font-bold text-medical-dark">XRay Vision AI</h1>
          </Link>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-muted-foreground hidden md:block">
            <span>Powered by ResNet50 V2</span>
          </div>
          
          <NavigationMenu>
            <NavigationMenuList className="flex gap-2">
              {!isExperimentPage && (
                <NavigationMenuItem>
                  <Link to="/experiment">
                    <Button 
                      size="sm" 
                      className="bg-medical hover:bg-medical-dark flex items-center gap-2"
                    >
                      <Plus className="h-4 w-4" />
                      <span className="hidden sm:inline">New Experiment</span>
                    </Button>
                  </Link>
                </NavigationMenuItem>
              )}
              {onShowHelp && (
                <NavigationMenuItem>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="flex items-center gap-2"
                    onClick={onShowHelp}
                  >
                    <HelpCircle className="h-4 w-4" />
                    <span className="hidden sm:inline">Help</span>
                  </Button>
                </NavigationMenuItem>
              )}
              <NavigationMenuItem>
                <Link to="/saved-experiments">
                  <Button variant="outline" size="sm" className="flex items-center gap-2">
                    <Save className="h-4 w-4" />
                    <span className="hidden sm:inline">Saved</span>
                  </Button>
                </Link>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </div>
    </header>
  );
};

export default Header;
