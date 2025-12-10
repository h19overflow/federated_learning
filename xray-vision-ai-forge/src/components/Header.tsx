import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Save, HelpCircle, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuList,
} from '@/components/ui/navigation-menu';

interface HeaderProps {
  onShowHelp?: () => void;
}

const Header = ({ onShowHelp }: HeaderProps = {}) => {
  const location = useLocation();
  const isExperimentPage = location.pathname === '/experiment';

  return (
    <header className="bg-white/80 backdrop-blur-xl border-b border-[hsl(210_15%_92%)] py-4 sticky top-0 z-50">
      <div className="container flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Link to="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity group">
            {/* Custom medical cross logo */}
            <div className="relative w-10 h-10 rounded-xl bg-[hsl(172_63%_22%)] flex items-center justify-center shadow-md shadow-[hsl(172_63%_22%)]/20 group-hover:shadow-lg group-hover:shadow-[hsl(172_63%_22%)]/30 transition-shadow">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none">
                <path d="M12 4v16M4 12h16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
              </svg>
            </div>
            <div className="flex flex-col">
              <h1 className="text-xl font-semibold text-[hsl(172_43%_15%)] tracking-tight">
                XRay Vision AI
              </h1>
              <span className="text-xs text-[hsl(215_15%_55%)] hidden sm:block">
                Medical Imaging Platform
              </span>
            </div>
          </Link>
        </div>

        <div className="flex items-center space-x-3">
          <div className="text-sm text-[hsl(215_15%_50%)] hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[hsl(168_25%_96%)] border border-[hsl(168_20%_90%)]">
            <div className="w-1.5 h-1.5 rounded-full bg-[hsl(152_60%_42%)]" />
            <span>ResNet50 V2</span>
          </div>

          <NavigationMenu>
            <NavigationMenuList className="flex gap-2">
              {!isExperimentPage && (
                <NavigationMenuItem>
                  <Link to="/experiment">
                    <Button
                      size="sm"
                      className="bg-[hsl(172_63%_22%)] hover:bg-[hsl(172_63%_18%)] flex items-center gap-2 rounded-xl shadow-sm hover:shadow-md transition-all"
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
                    className="flex items-center gap-2 text-[hsl(215_15%_40%)] hover:text-[hsl(172_63%_22%)] hover:bg-[hsl(168_25%_96%)] rounded-xl"
                    onClick={onShowHelp}
                  >
                    <HelpCircle className="h-4 w-4" />
                    <span className="hidden sm:inline">Help</span>
                  </Button>
                </NavigationMenuItem>
              )}
              <NavigationMenuItem>
                <Link to="/saved-experiments">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2 border-[hsl(210_15%_88%)] text-[hsl(172_43%_20%)] hover:bg-[hsl(168_25%_96%)] hover:border-[hsl(172_30%_80%)] rounded-xl transition-all"
                  >
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
