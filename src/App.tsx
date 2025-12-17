import { useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { ProjectsPage } from './components/ProjectsPage';
import { ProjectDashboard } from './components/ProjectDashboard';
import { SubjectDetail } from './components/SubjectDetail';
import { Info, X } from '@phosphor-icons/react';
import { Card } from './components/ui/card';

type View = 
  | { type: 'projects' }
  | { type: 'project'; projectId: string }
  | { type: 'subject'; subjectId: string };

function App() {
  const [view, setView] = useState<View>({ type: 'projects' });
  const [showInfo, setShowInfo] = useState(true);

  const handleSelectProject = (projectId: string) => {
    setView({ type: 'project', projectId });
  };

  const handleBackToProjects = () => {
    setView({ type: 'projects' });
  };

  const handleSelectSubject = (subjectId: string) => {
    setView({ type: 'subject', subjectId });
  };

  const handleBackToDashboard = () => {
    if (view.type === 'subject') {
      const projectId = localStorage.getItem('currentProjectId');
      if (projectId) {
        setView({ type: 'project', projectId });
      } else {
        setView({ type: 'projects' });
      }
    }
  };

  return (
    <>
      <div className="relative">
        {showInfo && (
          <Card className="fixed bottom-6 right-6 z-50 w-96 border-accent bg-card p-4 shadow-xl">
            <div className="flex items-start gap-3">
              <Info size={24} weight="fill" className="flex-shrink-0 text-accent" />
              <div className="flex-1">
                <h3 className="mb-1 font-semibold">Demo Mode</h3>
                <p className="text-sm text-muted-foreground">
                  This is a UI prototype demonstrating p-brain Local Studio's interface. In production, all computation
                  would run locally via a Python engine with no external network calls.
                </p>
              </div>
              <button
                onClick={() => setShowInfo(false)}
                className="flex-shrink-0 text-muted-foreground hover:text-foreground"
              >
                <X size={20} />
              </button>
            </div>
          </Card>
        )}

        {view.type === 'projects' && <ProjectsPage onSelectProject={handleSelectProject} />}
        
        {view.type === 'project' && (
          <>
            {(() => { localStorage.setItem('currentProjectId', view.projectId); return null; })()}
            <ProjectDashboard
              projectId={view.projectId}
              onBack={handleBackToProjects}
              onSelectSubject={handleSelectSubject}
            />
          </>
        )}
        
        {view.type === 'subject' && (
          <SubjectDetail
            subjectId={view.subjectId}
            onBack={handleBackToDashboard}
          />
        )}
      </div>
      <Toaster />
    </>
  );
}

export default App