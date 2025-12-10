import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  ArrowRight, 
  Brain, 
  Users, 
  Lock, 
  Zap, 
  TrendingUp,
  Server,
  Shield,
  Network,
  Database,
  Activity,
  CheckCircle2,
  Stethoscope,
  Settings
} from 'lucide-react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

const Landing = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/experiment');
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-gradient-to-b from-white to-blue-50">
      <Header />

      <main className="flex-1 overflow-y-auto">
        {/* Hero Section */}
        <section className="py-20 px-4">
          <div className="container max-w-6xl mx-auto text-center">
            <Badge className="mb-4 bg-medical text-white px-4 py-2 text-sm">
              Powered by ResNet50 V2 & Flower Framework
            </Badge>
            <h1 className="text-5xl md:text-6xl font-bold text-medical-dark mb-6">
              XRay Vision AI Forge
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Train state-of-the-art pneumonia detection models using <strong>Centralized</strong> or <strong>Federated Learning</strong>
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                className="bg-medical hover:bg-medical-dark text-lg px-8 py-6"
                onClick={handleGetStarted}
              >
                Start Training <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                className="text-lg px-8 py-6"
                onClick={() => document.getElementById('comparison')?.scrollIntoView({ behavior: 'smooth' })}
              >
                Learn More
              </Button>
            </div>
          </div>
        </section>

        {/* Features Grid */}
        <section className="py-16 px-4 bg-white">
          <div className="container max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12 text-medical-dark">
              Why Choose Our Platform?
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="border-2 hover:border-medical transition-all hover:shadow-lg">
                <CardHeader>
                  <Brain className="h-12 w-12 text-medical mb-4" />
                  <CardTitle>Advanced AI Models</CardTitle>
                  <CardDescription>
                    Pre-trained ResNet50 V2 architecture fine-tuned for medical imaging
                  </CardDescription>
                </CardHeader>
              </Card>
              <Card className="border-2 hover:border-medical transition-all hover:shadow-lg">
                <CardHeader>
                  <Shield className="h-12 w-12 text-medical mb-4" />
                  <CardTitle>Privacy-Preserving</CardTitle>
                  <CardDescription>
                    Federated learning keeps patient data private and secure on local devices
                  </CardDescription>
                </CardHeader>
              </Card>
              <Card className="border-2 hover:border-medical transition-all hover:shadow-lg">
                <CardHeader>
                  <Activity className="h-12 w-12 text-medical mb-4" />
                  <CardTitle>Real-Time Monitoring</CardTitle>
                  <CardDescription>
                    Watch training metrics and performance live with interactive visualizations
                  </CardDescription>
                </CardHeader>
              </Card>
            </div>
          </div>
        </section>

        {/* Comparison Section */}
        <section id="comparison" className="py-20 px-4 bg-gradient-to-b from-blue-50 to-white">
          <div className="container max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-medical-dark mb-4">
                Centralized vs Federated Learning
              </h2>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                Choose the training approach that best fits your needs
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
              {/* Centralized Learning Card */}
              <Card className="border-2 border-blue-200 hover:shadow-xl transition-all">
                <CardHeader className="bg-blue-50">
                  <div className="flex items-center justify-between mb-4">
                    <CardTitle className="text-2xl flex items-center gap-3">
                      <Server className="h-8 w-8 text-blue-600" />
                      Centralized Learning
                    </CardTitle>
                    <Badge variant="outline" className="bg-white">Traditional</Badge>
                  </div>
                  <CardDescription className="text-base">
                    All data is collected in one location for training
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  {/* Centralized Diagram */}
                  <div className="bg-white p-6 rounded-lg border-2 border-blue-200 mb-6">
                    <div className="flex flex-col items-center space-y-6">
                      {/* Multiple Clients */}
                      <div className="flex justify-center gap-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="flex flex-col items-center">
                            <Database className="h-10 w-10 text-blue-400" />
                            <span className="text-xs text-muted-foreground mt-1">Client {i}</span>
                          </div>
                        ))}
                      </div>
                      
                      {/* Arrows pointing down */}
                      <div className="flex justify-center gap-12">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="flex flex-col items-center">
                            <div className="h-12 w-0.5 bg-blue-400"></div>
                            <ArrowRight className="h-5 w-5 text-blue-400 rotate-90" />
                          </div>
                        ))}
                      </div>

                      {/* Central Server */}
                      <div className="relative">
                        <div className="absolute -inset-4 bg-blue-100 rounded-lg"></div>
                        <div className="relative bg-white p-4 rounded-lg border-2 border-blue-600 shadow-lg">
                          <div className="flex flex-col items-center">
                            <Server className="h-16 w-16 text-blue-600 mb-2" />
                            <span className="font-semibold text-blue-900">Central Server</span>
                            <span className="text-xs text-muted-foreground">All Data + Model</span>
                          </div>
                        </div>
                      </div>

                      {/* Training indicator */}
                      <div className="flex items-center gap-2 text-blue-600">
                        <Zap className="h-5 w-5" />
                        <span className="text-sm font-medium">Training Happens Here</span>
                      </div>
                    </div>
                  </div>

                  {/* Pros and Cons */}
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-green-700 mb-2 flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5" />
                        Advantages
                      </h4>
                      <ul className="space-y-1 text-sm text-muted-foreground ml-7">
                        <li>• Faster training time</li>
                        <li>• Simpler implementation</li>
                        <li>• Easier to debug</li>
                        <li>• Better for small datasets</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold text-orange-700 mb-2 flex items-center gap-2">
                        <Lock className="h-5 w-5" />
                        Considerations
                      </h4>
                      <ul className="space-y-1 text-sm text-muted-foreground ml-7">
                        <li>• Requires data centralization</li>
                        <li>• Privacy concerns with sensitive data</li>
                        <li>• Single point of failure</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Federated Learning Card */}
              <Card className="border-2 border-purple-200 hover:shadow-xl transition-all">
                <CardHeader className="bg-purple-50">
                  <div className="flex items-center justify-between mb-4">
                    <CardTitle className="text-2xl flex items-center gap-3">
                      <Users className="h-8 w-8 text-purple-600" />
                      Federated Learning
                    </CardTitle>
                    <Badge variant="outline" className="bg-white">Privacy-First</Badge>
                  </div>
                  <CardDescription className="text-base">
                    Training happens on local devices, only model updates are shared
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-6">
                  {/* Federated Diagram */}
                  <div className="bg-white p-6 rounded-lg border-2 border-purple-200 mb-6">
                    <div className="flex flex-col items-center space-y-4">
                      {/* Central Server at top */}
                      <div className="relative">
                        <div className="absolute -inset-4 bg-purple-100 rounded-lg"></div>
                        <div className="relative bg-white p-4 rounded-lg border-2 border-purple-600 shadow-lg">
                          <div className="flex flex-col items-center">
                            <Network className="h-16 w-16 text-purple-600 mb-2" />
                            <span className="font-semibold text-purple-900">Global Server</span>
                            <span className="text-xs text-muted-foreground">Aggregates Updates</span>
                          </div>
                        </div>
                      </div>

                      {/* Bidirectional arrows */}
                      <div className="flex justify-center gap-12">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="flex flex-col items-center gap-1">
                            <ArrowRight className="h-4 w-4 text-purple-400 rotate-90" />
                            <div className="h-8 w-0.5 bg-purple-300"></div>
                            <ArrowRight className="h-4 w-4 text-purple-400 -rotate-90" />
                          </div>
                        ))}
                      </div>

                      {/* Multiple Clients with local training */}
                      <div className="flex justify-center gap-4">
                        {[1, 2, 3].map((i) => (
                          <div key={i} className="relative">
                            <div className="bg-purple-50 p-3 rounded-lg border border-purple-300">
                              <div className="flex flex-col items-center">
                                <Database className="h-10 w-10 text-purple-500 mb-1" />
                                <Zap className="h-4 w-4 text-purple-600 mb-1" />
                                <span className="text-xs font-medium">Client {i}</span>
                                <span className="text-[10px] text-muted-foreground">Local Data</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Privacy indicator */}
                      <div className="flex items-center gap-2 text-purple-600">
                        <Shield className="h-5 w-5" />
                        <span className="text-sm font-medium">Data Stays Private</span>
                      </div>
                    </div>
                  </div>

                  {/* Pros and Cons */}
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold text-green-700 mb-2 flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5" />
                        Advantages
                      </h4>
                      <ul className="space-y-1 text-sm text-muted-foreground ml-7">
                        <li>• Data privacy preserved</li>
                        <li>• HIPAA/GDPR compliant</li>
                        <li>• Distributed computation</li>
                        <li>• Scalable to many clients</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold text-orange-700 mb-2 flex items-center gap-2">
                        <Lock className="h-5 w-5" />
                        Considerations
                      </h4>
                      <ul className="space-y-1 text-sm text-muted-foreground ml-7">
                        <li>• Longer training time</li>
                        <li>• More complex setup</li>
                        <li>• Network communication overhead</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Comparison Table */}
            <Card className="border-2">
              <CardHeader>
                <CardTitle className="text-center">Quick Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Feature</th>
                        <th className="px-6 py-3 text-center text-sm font-semibold text-blue-900">Centralized</th>
                        <th className="px-6 py-3 text-center text-sm font-semibold text-purple-900">Federated</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Data Privacy</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">⚠️ Data must be shared</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Data stays local</td>
                      </tr>
                      <tr className="bg-gray-50">
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Training Speed</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Fast</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">⚠️ Slower</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Setup Complexity</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Simple</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">⚠️ Complex</td>
                      </tr>
                      <tr className="bg-gray-50">
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Compliance (HIPAA/GDPR)</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">⚠️ Requires safeguards</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Built-in privacy</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Model Quality</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Excellent</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Comparable</td>
                      </tr>
                      <tr className="bg-gray-50">
                        <td className="px-6 py-4 text-sm font-medium text-gray-900">Scalability</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">⚠️ Limited by server</td>
                        <td className="px-6 py-4 text-center text-sm text-gray-600">✅ Highly scalable</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* How It Works */}
        <section className="py-20 px-4 bg-white">
          <div className="container max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12 text-medical-dark">
              How It Works
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { icon: Database, title: 'Upload Dataset', desc: 'Upload your chest X-ray images' },
                { icon: Settings, title: 'Configure', desc: 'Choose training mode and parameters' },
                { icon: Zap, title: 'Train', desc: 'Watch your model learn in real-time' },
                { icon: TrendingUp, title: 'Analyze', desc: 'Review metrics and performance' },
              ].map((step, index) => (
                <div key={index} className="relative">
                  <Card className="text-center hover:shadow-lg transition-all">
                    <CardContent className="pt-8 pb-6">
                      <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-medical text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">
                        {index + 1}
                      </div>
                      <step.icon className="h-12 w-12 text-medical mx-auto mb-4" />
                      <h3 className="font-semibold text-lg mb-2">{step.title}</h3>
                      <p className="text-sm text-muted-foreground">{step.desc}</p>
                    </CardContent>
                  </Card>
                  {index < 3 && (
                    <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2">
                      <ArrowRight className="h-6 w-6 text-medical" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4 bg-gradient-to-r from-medical to-medical-dark text-white">
          <div className="container max-w-4xl mx-auto text-center">
            <Stethoscope className="h-16 w-16 mx-auto mb-6" />
            <h2 className="text-4xl font-bold mb-4">
              Ready to Train Your AI Model?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Start detecting pneumonia with state-of-the-art machine learning
            </p>
            <Button 
              size="lg" 
              variant="secondary"
              className="text-lg px-8 py-6"
              onClick={handleGetStarted}
            >
              Get Started Now <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Landing;

