import { Component, type ErrorInfo, type ReactNode } from 'react'

type ErrorBoundaryState = { error: Error | null }

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('MOBA-RL arena render failed', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <main className="render-error" role="alert">
          <div className="eyebrow">ARENA DISPLAY ERROR</div>
          <h1>The match could not be displayed</h1>
          <p>{this.state.error.message || 'An unexpected rendering error occurred.'}</p>
          <button className="primary-button" onClick={() => window.location.reload()}>
            Reload arena
          </button>
        </main>
      )
    }

    return this.props.children
  }
}
