/**
 * Metadata Hooks
 * React Query hooks for reference data
 */

import { useQuery } from '@tanstack/react-query'
import { metadataService } from '../services/api'

export const QUERY_KEYS = {
  regions: 'metadata-regions',
  countries: 'metadata-countries',
  sites: 'metadata-sites',
  subjects: 'metadata-subjects',
  visitTypes: 'metadata-visit-types',
  queryTypes: 'metadata-query-types',
  userRoles: 'metadata-user-roles',
  studies: 'metadata-studies',
  formTypes: 'metadata-form-types',
  labTypes: 'metadata-lab-types',
  cras: 'metadata-cras',
}

export const useRegions = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.regions],
    queryFn: () => metadataService.getRegions(),
    staleTime: 30 * 60 * 1000, // 30 minutes - rarely changes
  })
}

export const useCountries = (regionId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.countries, regionId],
    queryFn: () => metadataService.getCountries(regionId),
    staleTime: 30 * 60 * 1000,
  })
}

export const useSites = (countryId?: string, regionId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.sites, countryId, regionId],
    queryFn: () => metadataService.getSites(countryId, regionId),
    staleTime: 15 * 60 * 1000,
  })
}

export const useSubjects = (siteId?: string) => {
  return useQuery({
    queryKey: [QUERY_KEYS.subjects, siteId],
    queryFn: () => metadataService.getSubjects(siteId),
    staleTime: 5 * 60 * 1000,
  })
}

export const useVisitTypes = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.visitTypes],
    queryFn: () => metadataService.getVisitTypes(),
    staleTime: 60 * 60 * 1000, // 1 hour
  })
}

export const useQueryTypes = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.queryTypes],
    queryFn: () => metadataService.getQueryTypes(),
    staleTime: 60 * 60 * 1000,
  })
}

export const useUserRoles = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.userRoles],
    queryFn: () => metadataService.getUserRoles(),
    staleTime: 60 * 60 * 1000,
  })
}

export const useStudies = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.studies],
    queryFn: () => metadataService.getStudies(),
    staleTime: 15 * 60 * 1000,
  })
}

export const useFormTypes = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.formTypes],
    queryFn: () => metadataService.getFormTypes(),
    staleTime: 60 * 60 * 1000,
  })
}

export const useLabTypes = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.labTypes],
    queryFn: () => metadataService.getLabTypes(),
    staleTime: 60 * 60 * 1000,
  })
}

export const useCRAs = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.cras],
    queryFn: () => metadataService.getCRAs(),
    staleTime: 15 * 60 * 1000,
  })
}
