import { Circle } from '@phosphor-icons/react';

interface ConnectionStatusProps {
  isConnected: boolean;
  className?: string;
}

export function ConnectionStatus({ isConnected, className = '' }: ConnectionStatusProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Circle
        size={8}
        weight="fill"
        className={`${isConnected ? 'text-success animate-pulse' : 'text-muted-foreground'}`}
      />
      <span className="text-xs text-muted-foreground">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );
}
