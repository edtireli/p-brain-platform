import { useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { ProjectsPage } from './components/ProjectsPage';
import { ProjectDashboard } from './components/ProjectDashboard';
import { SubjectDetail } from './components/SubjectDetail';
import { JobsPage } from './components/JobsPage';
import { LoginPage } from './components/LoginPage';
import { useSupabaseAuth } from '@/hooks/use-supabase-auth';
import { Button } from '@/components/ui/button';
import { supabase } from '@/lib/supabase';

type View = 
  | { type: 'projects' }
  | { type: 'project'; projectId: string }
  | { type: 'subject'; subjectId: string }
  | { type: 'jobs' };

function App() {
  const [view, setView] = useState<View>({ type: 'projects' });
  const auth = useSupabaseAuth();

  const showAuthPanel = auth.configured && !!auth.user;
  const authPanelTopClass = view.type === 'projects' ? 'top-4' : 'top-24';

  // Login is always the entry point.
  // - If Supabase isn't configured, show the login page with config instructions.
  // - If it is configured, require an active session.
  const needsLogin = !auth.configured || (!auth.loading && !auth.session);

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
    } else if (view.type === 'jobs') {
      setView({ type: 'projects' });
    }
  };

  return (
    <>
      <div className="relative">
        {needsLogin ? (
          <LoginPage />
        ) : (
          <>
            {showAuthPanel ? (
              <div className={`pointer-events-none fixed right-4 ${authPanelTopClass} z-50`}>
                <div className="pointer-events-auto flex flex-col items-end gap-2 rounded-md border border-border bg-background/80 px-3 py-2 text-xs text-muted-foreground backdrop-blur sm:flex-row sm:items-center">
                  <span className="mono max-w-[240px] truncate sm:max-w-[320px]">{auth.user.email}</span>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => supabase?.auth.signOut()}
                  >
                    Sign out
                  </Button>
                </div>
              </div>
            ) : null}

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

        {view.type === 'jobs' && (
          <JobsPage onBack={handleBackToDashboard} />
        )}
          </>
        )}
      </div>
      <Toaster />
    </>
  );
}

export default App