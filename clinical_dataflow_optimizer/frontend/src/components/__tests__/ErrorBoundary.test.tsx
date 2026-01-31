import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ErrorBoundary from '../ErrorBoundary'

const Boom = () => {
  throw new Error('Render crash')
}

describe('ErrorBoundary', () => {
  it('renders a visible error state instead of a white screen', () => {
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    )

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText(/Render crash/)).toBeInTheDocument()
  })
})
