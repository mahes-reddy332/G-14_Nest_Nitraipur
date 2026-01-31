import dayjs from 'dayjs'
import type { DateRangeFilter } from '../types'

export function getDaysFromRange(range: DateRangeFilter, fallbackDays: number): number {
  if (!range.start || !range.end) {
    return fallbackDays
  }
  const start = dayjs(range.start)
  const end = dayjs(range.end)
  const diff = Math.max(1, end.diff(start, 'day') + 1)
  return Math.min(365, diff)
}
