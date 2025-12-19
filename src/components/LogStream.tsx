import { useEffect, useRef, useState } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { engine } from '@/lib/engine';

interface LogStreamProps {
  jobId: string;
  className?: string;
}

export function LogStream({ jobId, className = '' }: LogStreamProps) {
  const [logs, setLogs] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  useEffect(() => {
    setLogs([]);
    
    const unsubscribe = engine.onJobLogs(jobId, (log) => {
      setLogs((prev) => [...prev, log]);
    });

    return () => {
      unsubscribe();
    };
  }, [jobId]);

  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    const isAtBottom = Math.abs(target.scrollHeight - target.scrollTop - target.clientHeight) < 10;
    autoScrollRef.current = isAtBottom;
  };

  const getLogColor = (log: string) => {
    if (log.includes('[ERROR]')) return 'text-destructive';
    if (log.includes('[SUCCESS]')) return 'text-success';
    if (log.includes('[WARNING]')) return 'text-warning';
    if (log.includes('[PROGRESS]')) return 'text-accent';
    return 'text-muted-foreground';
  };

  return (
    <div className={className}>
      <ScrollArea className="h-64 rounded-md border border-border bg-muted/30">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="mono p-4 text-xs"
        >
          {logs.length === 0 ? (
            <div className="text-muted-foreground">Waiting for logs...</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className={`mb-1 ${getLogColor(log)}`}>
                {log}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
