/**
 * Error Boundary Component
 * 
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI instead of crashing.
 * 
 * GitHub Issue #2: Error Boundaries
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showDetails?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Update state with error details
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Send error to logging service (if configured)
    this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // Create error report
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      userId: null, // TODO: Get from auth context if available
      sessionId: sessionStorage.getItem('sessionId') || null,
      additionalContext: {
        errorName: error.name,
        errorToString: error.toString(),
      },
    };

    // Log to console for development
    console.error('Error Report:', JSON.stringify(errorReport, null, 2));

    // Send to backend logging endpoint
    fetch('/api/v1/errors/log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(errorReport),
    })
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          console.log(`Error logged successfully with ID: ${data.error_id}`);
        } else {
          console.warn('Error logging failed:', data.message);
        }
      })
      .catch(err => {
        // Don't fail if logging fails - just log to console
        console.error('Failed to send error report to backend:', err);
      });
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
          <Card className="max-w-2xl w-full">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
                  <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
                </div>
                <div>
                  <CardTitle className="text-xl">Something went wrong</CardTitle>
                  <CardDescription>
                    An unexpected error occurred. We've logged the issue and will look into it.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {this.props.showDetails && this.state.error && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                    Error Details:
                  </h3>
                  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-auto max-h-60">
                    <p className="text-sm font-mono text-red-600 dark:text-red-400 mb-2">
                      {this.state.error.message}
                    </p>
                    {this.state.error.stack && (
                      <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                        {this.state.error.stack}
                      </pre>
                    )}
                  </div>

                  {this.state.errorInfo?.componentStack && (
                    <>
                      <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300 mt-4">
                        Component Stack:
                      </h3>
                      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-auto max-h-40">
                        <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                          {this.state.errorInfo.componentStack}
                        </pre>
                      </div>
                    </>
                  )}
                </div>
              )}

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                <h3 className="font-semibold text-sm text-blue-900 dark:text-blue-100 mb-2">
                  What you can do:
                </h3>
                <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1 list-disc list-inside">
                  <li>Try refreshing the page</li>
                  <li>Go back to the home page</li>
                  <li>Clear your browser cache and cookies</li>
                  <li>Contact support if the problem persists</li>
                </ul>
              </div>
            </CardContent>

            <CardFooter className="flex gap-3">
              <Button
                onClick={this.handleReset}
                variant="default"
                className="flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Try Again
              </Button>
              <Button
                onClick={this.handleGoHome}
                variant="outline"
                className="flex items-center gap-2"
              >
                <Home className="h-4 w-4" />
                Go Home
              </Button>
            </CardFooter>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;


/**
 * Usage Examples:
 * 
 * 1. Wrap entire app:
 * ```tsx
 * <ErrorBoundary>
 *   <App />
 * </ErrorBoundary>
 * ```
 * 
 * 2. Wrap specific sections:
 * ```tsx
 * <ErrorBoundary showDetails={true}>
 *   <Dashboard />
 * </ErrorBoundary>
 * ```
 * 
 * 3. Custom fallback:
 * ```tsx
 * <ErrorBoundary fallback={<CustomErrorPage />}>
 *   <ComplexComponent />
 * </ErrorBoundary>
 * ```
 * 
 * 4. With error handler:
 * ```tsx
 * <ErrorBoundary 
 *   onError={(error, errorInfo) => {
 *     // Send to analytics
 *     analytics.track('error', { error, errorInfo });
 *   }}
 * >
 *   <App />
 * </ErrorBoundary>
 * ```
 */

