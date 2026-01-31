/**
 * Metadata Service
 * Handles regions, countries, sites, and other reference data
 */

import { apiClient } from './client'
import type { Region, Country, Site } from './types'

export class MetadataService {
  private basePath = '/metadata'

  async getRegions(): Promise<Region[]> {
    return apiClient.get<Region[]>(`${this.basePath}/regions`)
  }

  async getCountries(regionId?: string): Promise<Country[]> {
    return apiClient.get<Country[]>(`${this.basePath}/countries`, {
      params: { regionId },
    })
  }

  async getSites(countryId?: string, regionId?: string): Promise<Site[]> {
    return apiClient.get<Site[]>(`${this.basePath}/sites`, {
      params: { countryId, regionId },
    })
  }

  async getSubjects(siteId?: string): Promise<{
    id: string
    subjectId: string
    siteId: string
    status: string
  }[]> {
    return apiClient.get(`${this.basePath}/subjects`, {
      params: { siteId },
    })
  }

  async getVisitTypes(): Promise<{
    id: string
    name: string
    code: string
    order: number
  }[]> {
    return apiClient.get(`${this.basePath}/visit-types`)
  }

  async getQueryTypes(): Promise<{
    id: string
    name: string
    code: string
    description: string
  }[]> {
    return apiClient.get(`${this.basePath}/query-types`)
  }

  async getUserRoles(): Promise<{
    id: string
    name: string
    permissions: string[]
  }[]> {
    return apiClient.get(`${this.basePath}/user-roles`)
  }

  async getStudies(): Promise<{
    id: string
    name: string
    protocol: string
    status: string
    startDate: string
    endDate?: string
  }[]> {
    return apiClient.get(`${this.basePath}/studies`)
  }

  async getFormTypes(): Promise<{
    id: string
    name: string
    code: string
    category: string
  }[]> {
    return apiClient.get(`${this.basePath}/form-types`)
  }

  async getLabTypes(): Promise<{
    id: string
    name: string
    code: string
    category: string
  }[]> {
    return apiClient.get(`${this.basePath}/lab-types`)
  }

  async getCRAs(): Promise<{
    id: string
    name: string
    email: string
    region: string
    assignedSites: string[]
  }[]> {
    return apiClient.get(`${this.basePath}/cras`)
  }
}

export const metadataService = new MetadataService()
