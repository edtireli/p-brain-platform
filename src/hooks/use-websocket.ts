import { useEffect, useRef, useState } from 'react';

export interface WebSocketMessage {
  type: 'job_progress' | 'job_log' | 'job_status' | 'subject_stage_status';
  payload: any;
}

export function useWebSocket(url: string, enabled: boolean = true) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set());

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let shouldReconnect = true;

    const connect = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        return;
      }

      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as WebSocketMessage;
            setLastMessage(message);
            messageHandlersRef.current.forEach(handler => handler(message));
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          setIsConnected(false);
          wsRef.current = null;

          if (shouldReconnect) {
            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, 3000);
          }
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        if (shouldReconnect) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      }
    };

    connect();

    return () => {
      shouldReconnect = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, enabled]);

  const send = (message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  const subscribe = (handler: (message: WebSocketMessage) => void) => {
    messageHandlersRef.current.add(handler);
    return () => {
      messageHandlersRef.current.delete(handler);
    };
  };

  return {
    isConnected,
    lastMessage,
    send,
    subscribe,
  };
}
