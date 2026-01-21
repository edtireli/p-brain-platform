import { useEffect, useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { ProjectsPage } from './components/ProjectsPage';
import { ProjectDashboard } from './components/ProjectDashboard';
import { SubjectDetail } from './components/SubjectDetail';
import { JobsPage } from './components/JobsPage';
import { OnboardingWizard } from './components/OnboardingWizard';
import { ProjectAnalysis } from './components/ProjectAnalysis';
import { engine } from '@/lib/engine';

type View = 
  | { type: 'projects' }
  | { type: 'project'; projectId: string }
  | { type: 'analysis'; projectId: string }
  | { type: 'subject'; subjectId: string }
  | { type: 'jobs' };

function App() {
  const [view, setView] = useState<View>({ type: 'projects' });
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingChecked, setOnboardingChecked] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await engine.getSettings();
        if (!cancelled) setShowOnboarding(!s.onboardingCompleted);
      } catch {
        // If backend isn't ready yet, don't block the UI.
        if (!cancelled) setShowOnboarding(false);
      } finally {
        if (!cancelled) setOnboardingChecked(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSelectProject = (projectId: string) => {
    setView({ type: 'project', projectId });
  };

  const handleBackToProjects = () => {
    setView({ type: 'projects' });
  };

  const handleSelectSubject = (subjectId: string) => {
    setView({ type: 'subject', subjectId });
  };

  const handleOpenAnalysis = (projectId: string) => {
    setView({ type: 'analysis', projectId });
  };

  const handleBackToDashboard = () => {
    if (view.type === 'subject') {
      const projectId = localStorage.getItem('currentProjectId');
      if (projectId) {
        setView({ type: 'project', projectId });
      } else {
        setView({ type: 'projects' });
      }
    } else if (view.type === 'analysis') {
      setView({ type: 'project', projectId: view.projectId });
    } else if (view.type === 'jobs') {
      setView({ type: 'projects' });
    }
  };

  return (
    <>
      <div className="min-h-screen">
        {onboardingChecked && showOnboarding ? (
          <OnboardingWizard
            onDone={() => {
              setShowOnboarding(false);
            }}
          />
        ) : null}
        {view.type === 'projects' && <ProjectsPage onSelectProject={handleSelectProject} />}

        {view.type === 'project' && (
          <>
            {(() => {
              localStorage.setItem('currentProjectId', view.projectId);
              return null;
            })()}
            <ProjectDashboard
              projectId={view.projectId}
              onBack={handleBackToProjects}
              onSelectSubject={handleSelectSubject}
              onOpenAnalysis={() => handleOpenAnalysis(view.projectId)}
            />
          </>
        )}

        {view.type === 'analysis' && (
          <ProjectAnalysis
            projectId={view.projectId}
            onBack={handleBackToDashboard}
          />
        )}

        {view.type === 'subject' && <SubjectDetail subjectId={view.subjectId} onBack={handleBackToDashboard} />}

        {view.type === 'jobs' && <JobsPage onBack={handleBackToDashboard} />}
      </div>
      <Toaster />
    </>
  );
}

export default App