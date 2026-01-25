import React, { useState, useEffect } from 'react'
import { Select, DatePicker, Button, Space, Typography, Badge } from 'antd'
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
  const { selectedStudyId, setSelectedStudyId, isConnected } = useStore()
  const [selectedSite, setSelectedSite] = useState<string | null>(null)
  const [dateRange, setDateRange] = useState<[Dayjs | null, Dayjs | null] | null>(null)

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

  const hasActiveFilters = selectedStudyId || selectedSite || dateRange

  const clearAllFilters = () => {
    setSelectedStudyId(null)
    setSelectedSite(null)
    setDateRange(null)
  }

  const activeFilterCount = [selectedStudyId, selectedSite, dateRange].filter(Boolean).length

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
          onChange={setSelectedStudyId}
          options={studies.map((study: any) => ({
            value: study.study_id,
            label: study.name || study.study_id,
          }))}
          showSearch
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
        />
      </div>

      {/* Site/Region Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <EnvironmentOutlined style={{ color: 'var(--gray-500)' }} />
        <Select
          placeholder="All Sites"
          style={{ minWidth: 160 }}
          allowClear
          value={selectedSite}
          onChange={setSelectedSite}
          disabled={!selectedStudyId}
          options={sites.map((site: any) => ({
            value: site.site_id,
            label: `${site.name || site.site_id} (${site.country || 'N/A'})`,
          }))}
          showSearch
          filterOption={(input, option) =>
            (option?.label as string)?.toLowerCase().includes(input.toLowerCase())
          }
        />
      </div>

      {/* Date Range */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <CalendarOutlined style={{ color: 'var(--gray-500)' }} />
        <RangePicker
          value={dateRange}
          onChange={(dates) => setDateRange(dates as [Dayjs | null, Dayjs | null] | null)}
          style={{ minWidth: 240 }}
          presets={[
            { label: 'Last 7 Days', value: [dayjs().subtract(7, 'd'), dayjs()] },
            { label: 'Last 30 Days', value: [dayjs().subtract(30, 'd'), dayjs()] },
            { label: 'Last 90 Days', value: [dayjs().subtract(90, 'd'), dayjs()] },
            { label: 'This Month', value: [dayjs().startOf('month'), dayjs()] },
            { label: 'This Quarter', value: [dayjs().subtract(3, 'month').startOf('month'), dayjs()] },
          ]}
        />
      </div>

      {/* Clear Filters */}
      {hasActiveFilters && (
        <Button 
          type="text" 
          icon={<ClearOutlined />} 
          onClick={clearAllFilters}
          style={{ color: 'var(--gray-500)' }}
        >
          Clear
        </Button>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Real-time indicator & Refresh */}
      <Space size={16}>
        {/* Connection Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <div 
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: isConnected ? 'var(--status-healthy)' : 'var(--status-critical)',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }}
          />
          <Text style={{ fontSize: 12, color: 'var(--gray-500)' }}>
            {isConnected ? 'Live' : 'Disconnected'}
          </Text>
        </div>

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
        >
          Refresh
        </Button>
      </Space>
    </div>
  )
}
