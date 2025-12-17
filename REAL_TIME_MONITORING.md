# Real-Time Job Monitoring Implementation

## Overview

The p-brain Local Studio implements real-time job monitoring through event-driven architecture that simulates WebSocket-style communication patterns. This provides users with immediate feedback on long-running computational tasks without polling overhead.

## Architecture

### Event System

The mock engine (`src/lib/mock-engine.ts`) implements a publish-subscribe pattern with three primary event streams:

1. **Job Updates** - Emitted when job status, progress, or metadata changes
2. **Status Updates** - Emitted when subject stage status changes (not_run → running → done/failed)
3. **Log Updates** - Emitted during job execution for real-time log streaming

### Components

#### 1. JobMonitorPanel (`src/components/JobMonitorPanel.tsx`)

Primary UI component for job monitoring featuring:

- **Live Job List**: Real-time updating list of all jobs with filtering by status
- **Job Statistics Cards**: Running, queued, completed, and failed job counts
- **Expandable Job Details**: Click to expand and view comprehensive job information
- **Action Controls**: Cancel running/queued jobs, retry failed/cancelled jobs
- **Status Indicators**: Visual badges and icons for job states
- **Progress Bars**: Live progress updates with percentage and current step display

Key features:
- Auto-refresh every 5 seconds for resilience
- Event-driven updates for immediate feedback
- Tabbed filtering (All, Running, Queued, Completed, Failed)
- Persistent expanded state per job

#### 2. LogStream (`src/components/LogStream.tsx`)

Real-time log display component with:

- **Color-Coded Logs**: Different colors for ERROR, SUCCESS, WARNING, PROGRESS, INFO
- **Auto-Scroll**: Automatically scrolls to bottom unless user manually scrolls
- **Event Subscription**: Subscribes to job-specific log streams
- **Clean Lifecycle**: Properly cleans up subscriptions on unmount

#### 3. ConnectionStatus (`src/components/ConnectionStatus.tsx`)

Visual indicator of the event system's connectivity:

- Animated pulse for connected state
- Clear visual feedback for disconnected state
- Compact design suitable for headers

#### 4. useWebSocket Hook (`src/hooks/use-websocket.ts`)

Reusable hook for WebSocket integration (prepared for production use):

- Auto-reconnection with exponential backoff
- Message handler subscription system
- Connection state management
- Type-safe message handling

## Data Flow

```
User Action (Run Pipeline)
    ↓
mockEngine.createJob()
    ↓
Job Created & Queued
    ↓
executeJob() begins
    ↓
┌─────────────────────────────────────┐
│  Real-Time Event Emissions:         │
│  1. Job status: queued → running    │
│  2. Stage status: not_run → running │
│  3. Progress updates (0-100%)       │
│  4. Log entries streamed            │
│  5. Job status: running → completed │
│  6. Stage status: running → done    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  UI Updates (via event listeners):  │
│  - JobMonitorPanel refreshes list   │
│  - Progress bars animate            │
│  - LogStream displays new lines     │
│  - Status badges update             │
│  - Dashboard grid updates           │
└─────────────────────────────────────┘
```

## Event Subscription Pattern

### Job Updates

```typescript
useEffect(() => {
  const unsubscribe = mockEngine.onJobUpdate(updatedJob => {
    setJobs(prevJobs => {
      const index = prevJobs.findIndex(j => j.id === updatedJob.id);
      if (index >= 0) {
        const newJobs = [...prevJobs];
        newJobs[index] = updatedJob;
        return newJobs;
      }
      return [updatedJob, ...prevJobs];
    });
  });

  return () => unsubscribe();
}, []);
```

### Status Updates

```typescript
useEffect(() => {
  const unsubscribe = mockEngine.onStatusUpdate(update => {
    setSubjects(prev =>
      prev.map(s =>
        s.id === update.subjectId
          ? { ...s, stageStatuses: { ...s.stageStatuses, [update.stageId]: update.status } }
          : s
      )
    );
  });

  return () => unsubscribe();
}, []);
```

### Log Streaming

```typescript
useEffect(() => {
  const unsubscribe = mockEngine.onJobLogs(jobId, (log) => {
    setLogs((prev) => [...prev, log]);
  });

  return () => unsubscribe();
}, [jobId]);
```

## Job States

### Status Transitions

```
queued → running → completed
              ↓
            failed
              ↓
          (retry) → queued
              ↓
          cancelled
```

### Visual Indicators

- **queued**: Clock icon, secondary badge
- **running**: Spinning spinner, accent badge, progress bar
- **completed**: Check icon, success badge
- **failed**: X icon, destructive badge, error message
- **cancelled**: X icon, muted badge

## Performance Considerations

### Efficient Updates

1. **Event-Driven**: Only updates when actual changes occur
2. **Scoped Subscriptions**: Job-specific log listeners prevent memory leaks
3. **Batched Rendering**: React batches state updates automatically
4. **Auto-Scroll Optimization**: Only scrolls when user at bottom

### Fallback Polling

The system combines event-driven updates with periodic polling (5s interval) to ensure:
- Recovery from missed events
- Consistency after page refresh
- Resilience to temporary disconnections

## Integration Points

### Dashboard Integration

The ProjectDashboard displays an active job count badge:

```typescript
const [activeJobsCount, setActiveJobsCount] = useState(0);

useEffect(() => {
  const interval = setInterval(async () => {
    const jobs = await mockEngine.getJobs({ projectId });
    const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
    setActiveJobsCount(active);
  }, 2000);

  return () => clearInterval(interval);
}, [projectId]);
```

### Subject Detail Integration

Subject pages subscribe to status updates to reflect changes in real-time:

```typescript
useEffect(() => {
  const unsubscribe = mockEngine.onStatusUpdate(update => {
    if (update.subjectId === subjectId) {
      loadSubject(); // Refresh subject data
    }
  });

  return () => unsubscribe();
}, [subjectId]);
```

## Production WebSocket Migration

When integrating with a real Python backend via WebSocket:

1. Replace mock engine event listeners with WebSocket message handlers
2. Use the `useWebSocket` hook for connection management
3. Map WebSocket message types to the existing event patterns:
   - `job_progress` → `onJobUpdate`
   - `job_status` → `onJobUpdate`
   - `job_log` → `onJobLogs`
   - `subject_stage_status` → `onStatusUpdate`

Example WebSocket integration:

```typescript
const { subscribe } = useWebSocket('ws://localhost:8000/ws');

useEffect(() => {
  const unsubscribe = subscribe((message) => {
    switch (message.type) {
      case 'job_progress':
        handleJobUpdate(message.payload);
        break;
      case 'job_log':
        handleLogEntry(message.payload);
        break;
      case 'subject_stage_status':
        handleStatusUpdate(message.payload);
        break;
    }
  });

  return () => unsubscribe();
}, []);
```

## Testing Scenarios

### Simulated Behaviors

The mock engine simulates realistic scenarios:

1. **Progressive Execution**: Jobs move through stages with realistic timing
2. **Random Failures**: 5% chance of failure at mid-point (configurable)
3. **Concurrent Jobs**: Multiple jobs execute in parallel
4. **Log Variety**: Different log types (INFO, PROGRESS, ERROR, SUCCESS)

### User Scenarios

Test these flows:

1. **Run Full Pipeline**: Start pipeline for multiple subjects, observe dashboard updates
2. **Monitor Active Jobs**: Open job monitor, watch progress bars and logs
3. **Cancel Job**: Cancel a running job, verify immediate stop
4. **Retry Failed Job**: Retry a failed job, verify new execution
5. **Filter Jobs**: Switch between status tabs, verify correct filtering
6. **Expand Job Details**: Click job card, view logs and metadata
7. **Simultaneous Updates**: Multiple jobs completing, verify no UI conflicts

## Future Enhancements

1. **Job Priority**: Allow users to prioritize certain jobs
2. **Job Dependencies**: Visual dependency graph for stage ordering
3. **Resource Monitoring**: CPU/memory usage per job
4. **Job Templates**: Save and reuse job configurations
5. **Notifications**: Browser notifications for job completion
6. **Job History**: Searchable history with performance analytics
7. **Batch Operations**: Cancel/retry multiple jobs at once
8. **Export Logs**: Download job logs as text files
