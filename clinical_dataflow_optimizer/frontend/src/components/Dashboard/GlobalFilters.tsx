import React, { useMemo, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { Select, DatePicker, Button, Space, Typography, Badge, Tag, Tooltip } from 'antd'
import {
  FilterOutlined,
  ReloadOutlined,
  EnvironmentOutlined,
  CalendarOutlined,
  ExperimentOutlined,
  ClearOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs, { Dayjs } from 'dayjs'
import { studiesApi } from '../../api'
import { useStore } from '../../store'
import '../../styles/clinical-design-system.css'

const { Text } = Typography
const { RangePicker } = DatePicker

interface GlobalFiltersProps {
  onRefresh?: () => void
  lastUpdated?: string
}

/**
 * GlobalFilters Component
 * 
 * Sticky filter bar with:
 * - Study selector
 * - Site/region filter
 * - Date range picker
 * - Clear all filters
 * - Last updated timestamp
 */
export default function GlobalFilters({ onRefresh, lastUpdated }: GlobalFiltersProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const {
    selectedStudyId,
    setSelectedStudyId,
    selectedSiteId,
    setSelectedSiteId,
    dateRange,
    setDateRange,
    isConnected,
    connectionState,
    connectionError,
  } = useStore()

  // Fetch available studies
  const { data: studies = [] } = useQuery({
    queryKey: ['studies'],
    queryFn: studiesApi.getAll,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })

  // Fetch sites for selected study
  const { data: sites = [] } = useQuery({
    queryKey: ['sites', selectedStudyId],
    queryFn: () => selectedStudyId ? studiesApi.getSites(selectedStudyId) : Promise.resolve([]),
    enabled: !!selectedStudyId,
  })

  const hasActiveFilters = selectedStudyId || selectedSiteId || dateRange.start || dateRange.end

  const clearAllFilters = () => {
    setSelectedStudyId(null)
    setSelectedSiteId(null)
    setDateRange({ start: null, end: null })
    if (location.pathname.startsWith('/studies')) {
      navigate('/studies')
    }
  }

  const activeFilterCount = [selectedStudyId, selectedSiteId, dateRange.start && dateRange.end].filter(Boolean).length

  const pickerRange: [Dayjs | null, Dayjs | null] | null = useMemo(() => {
    if (dateRange.start && dateRange.end) {
      return [dayjs(dateRange.start), dayjs(dateRange.end)]
    }
    return null
  }, [dateRange])

  const studyOptions = useMemo(
    () => studies.map((study: any) => ({
      value: study.study_id,
      label: study.name || study.study_id,
    })),
    [studies]
  )

  const siteOptions = useMemo(
    () => sites.map((site: any) => ({
      value: site.site_id,
      label: `${site.name || site.site_id} (${site.country || 'N/A'})`,
    })),
    [sites]
  )

  useEffect(() => {
    setSelectedSiteId(null)
  }, [selectedStudyId, setSelectedSiteId])

  const routeStudyId = useMemo(() => {
    const match = location.pathname.match(/^\/studies\/([^/]+)$/)
    return match ? decodeURIComponent(match[1]) : null
  }, [location.pathname])

  useEffect(() => {
    if (!location.pathname.startsWith('/studies')) return

    if (selectedStudyId && selectedStudyId !== routeStudyId) {
      navigate(`/studies/${selectedStudyId}`)
      return
    }

    if (!selectedStudyId && routeStudyId) {
      navigate('/studies')
    }
  }, [selectedStudyId, routeStudyId, location.pathname, navigate])

  const connectionLabel = isConnected
    ? 'Connected'
    : connectionState === 'retrying'
      ? 'Retrying'
      : connectionState === 'waiting_for_backend'
        ? 'Starting'
        : connectionState === 'failed'
          ? 'Failed'
          : 'Disconnected'

  return (
    <div className="global-filters">
      {/* Filter Icon */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <FilterOutlined style={{ fontSize: 16, color: 'var(--clinical-primary)' }} />
        <Text className="global-filters__label">Filters</Text>
        {activeFilterCount > 0 && (
          <Badge 
            count={activeFilterCount} 
            style={{ backgroundColor: 'var(--clinical-primary)' }}
          />
        )}
      </div>

      <div className="global-filters__divider" />

      {/* Study Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <ExperimentOutlined style={{ color: 'var(--gray-500)' }} />
        <Select
          placeholder="All Studies"
          style={{ minWidth: 180 }}
          allowClear
          value={selectedStudyId}
          onChange={(value) => {
            setSelectedStudyId(value ?? null)
            if (location.pathname.startsWith('/studies')) {
              navigate(value ? `/studies/${value}` : '/studies')
            }
          }}
          options={studyOptions}
          showSearch
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
          aria-label="Filter by study"
        />
      </div>

      {/* Site/Region Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <EnvironmentOutlined style={{ color: 'var(--gray-500)' }} />
        <Select
          placeholder="All Sites"
          style={{ minWidth: 160 }}
          allowClear
          value={selectedSiteId}
          onChange={setSelectedSiteId}
          disabled={!selectedStudyId}
          options={siteOptions}
          showSearch
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
          aria-label="Filter by site"
        />
      </div>

      {/* Date Range */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <CalendarOutlined style={{ color: 'var(--gray-500)' }} />
        <RangePicker
          value={pickerRange}
          onChange={(dates) => {
            const [start, end] = (dates || []) as [Dayjs | null, Dayjs | null]
            setDateRange({
              start: start ? start.startOf('day').toISOString() : null,
              end: end ? end.endOf('day').toISOString() : null,
            })
          }}
          style={{ minWidth: 240 }}
          presets={[
            { label: 'Last 7 Days', value: [dayjs().subtract(7, 'd'), dayjs()] },
            { label: 'Last 30 Days', value: [dayjs().subtract(30, 'd'), dayjs()] },
            { label: 'Last 90 Days', value: [dayjs().subtract(90, 'd'), dayjs()] },
            { label: 'This Month', value: [dayjs().startOf('month'), dayjs()] },
            { label: 'This Quarter', value: [dayjs().subtract(3, 'month').startOf('month'), dayjs()] },
          ]}
          aria-label="Filter by date range"
        />
      </div>

      {/* Clear Filters */}
      {hasActiveFilters && (
        <Button 
          type="text" 
          icon={<ClearOutlined />} 
          onClick={clearAllFilters}
          style={{ color: 'var(--gray-500)' }}
          aria-label="Clear all filters"
        >
          Clear
        </Button>
      )}

      {/* Applied Filters */}
      {hasActiveFilters && (
        <div className="global-filters__applied">
          {selectedStudyId && (
            <Tag className="filter-chip" closable onClose={() => setSelectedStudyId(null)}>
              Study: {selectedStudyId}
            </Tag>
          )}
          {selectedSiteId && (
            <Tag className="filter-chip" closable onClose={() => setSelectedSiteId(null)}>
              Site: {selectedSiteId}
            </Tag>
          )}
          {dateRange.start && dateRange.end && (
            <Tag className="filter-chip" closable onClose={() => setDateRange({ start: null, end: null })}>
              Dates: {dayjs(dateRange.start).format('MMM D')}–{dayjs(dateRange.end).format('MMM D')}
            </Tag>
          )}
        </div>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Real-time indicator & Refresh */}
      <Space size={16}>
        {/* Connection Status */}
        <Tooltip title={connectionError || 'Real-time connection status'}>
          <div className={`connection-pill connection-pill--${isConnected ? 'connected' : connectionState}`}>
            <span className="connection-pill__dot" />
            <Text style={{ fontSize: 12, color: 'var(--gray-600)' }}>{connectionLabel}</Text>
          </div>
        </Tooltip>

        {/* Last Updated */}
        {lastUpdated && (
          <Text style={{ fontSize: 12, color: 'var(--gray-400)' }}>
            Updated: {dayjs(lastUpdated).format('HH:mm:ss')}
          </Text>
        )}

        {/* Refresh Button */}
        <Button
          type="text"
          icon={<ReloadOutlined />}
          onClick={onRefresh}
          style={{ color: 'var(--clinical-primary)' }}
          aria-label="Refresh dashboard data"
        >
          Refresh
        </Button>
      </Space>
    </div>
  )
}
