import { lazy } from 'react'
import {
    DashboardOutlined,
    UserOutlined,
    ExperimentOutlined,
    SafetyOutlined,
    MedicineBoxOutlined,
    ScheduleOutlined,
    FormOutlined,
    TeamOutlined,
    FileTextOutlined,
    DatabaseOutlined,
    AlertOutlined,
    RobotOutlined,
    MessageOutlined,
    CheckCircleOutlined,
} from '@ant-design/icons'

// Lazy load pages
const Dashboard = lazy(() => import('../pages/Dashboard'))
const Studies = lazy(() => import('../pages/Studies'))
const Patients = lazy(() => import('../pages/Patients'))
const Sites = lazy(() => import('../pages/Sites'))
const EDCMetrics = lazy(() => import('../pages/EDCMetrics'))
const DataQuality = lazy(() => import('../pages/DataQuality'))
const VisitManagement = lazy(() => import('../pages/VisitManagement'))
const LaboratoryData = lazy(() => import('../pages/LaboratoryData'))
const SafetyMonitoring = lazy(() => import('../pages/SafetyMonitoring'))
const MedDRACoding = lazy(() => import('../pages/MedDRACoding'))
const WHODrugCoding = lazy(() => import('../pages/WHODrugCoding'))
const FormsVerification = lazy(() => import('../pages/FormsVerification'))
const CRAActivity = lazy(() => import('../pages/CRAActivity'))
const Reports = lazy(() => import('../pages/Reports'))
const Alerts = lazy(() => import('../pages/Alerts'))
const Agents = lazy(() => import('../pages/Agents'))
const Conversational = lazy(() => import('../pages/Conversational'))

export interface RouteConfig {
    path: string
    key: string
    label?: string
    icon?: React.ReactNode
    element?: React.LazyExoticComponent<any> | React.ComponentType
    children?: RouteConfig[]
    redirect?: string // If this route is just a redirect
    hideInMenu?: boolean
}

export const routes: RouteConfig[] = [
    {
        path: '/dashboard',
        key: '/dashboard',
        label: 'Dashboard Home',
        icon: <DashboardOutlined />,
        element: Dashboard,
    },
    {
        path: '/patient-site', // Virtual parent for menu
        key: 'patient-site',
        label: 'Patient & Site Metrics',
        icon: <UserOutlined />,
        children: [
            {
                path: '/patients',
                key: '/patients',
                label: 'Patient Metrics',
                element: Patients,
            },
            {
                path: '/sites',
                key: '/sites',
                label: 'Site Metrics',
                element: Sites,
            },
            {
                path: '/edc-metrics',
                key: '/edc-metrics',
                label: 'EDC Metrics',
                element: EDCMetrics,
            },
            // Dynamic parameters need to be handled carefully in menu vs router
            // We'll define them but hide from menu
            {
                path: '/patients/:patientId',
                key: '/patients/:patientId',
                element: Patients,
                hideInMenu: true,
            },
            {
                path: '/sites/:siteId',
                key: '/sites/:siteId',
                element: Sites,
                hideInMenu: true,
            },
        ],
    },
    {
        path: '/data-quality',
        key: '/data-quality',
        label: 'Data Quality',
        icon: <CheckCircleOutlined />,
        element: DataQuality,
    },
    {
        path: '/visit-management',
        key: '/visit-management',
        label: 'Visit Management',
        icon: <ScheduleOutlined />,
        element: VisitManagement,
    },
    {
        path: '/laboratory',
        key: '/laboratory',
        label: 'Laboratory Data',
        icon: <ExperimentOutlined />,
        element: LaboratoryData,
    },
    // Alias for legacy support
    {
        path: '/laboratory-data',
        key: '/laboratory-data',
        element: LaboratoryData,
        hideInMenu: true,
    },
    {
        path: '/safety-monitoring',
        key: '/safety-monitoring',
        label: 'Safety Monitoring',
        icon: <SafetyOutlined />,
        element: SafetyMonitoring,
    },
    {
        path: '/coding',
        key: 'coding', // Parent key
        label: 'Coding & Reconciliation',
        icon: <MedicineBoxOutlined />,
        // For router, if someone hits /coding, redirect to first child
        redirect: '/coding/meddra',
        children: [
            {
                path: '/coding/meddra',
                key: '/coding/meddra',
                label: 'MedDRA Coding',
                element: MedDRACoding,
            },
            {
                path: '/coding/whodrug',
                key: '/coding/whodrug',
                label: 'WHO Drug Coding',
                element: WHODrugCoding,
            },
            // Legacy redirects
            {
                path: '/meddra-coding',
                key: '/meddra-coding',
                redirect: '/coding/meddra',
                hideInMenu: true,
            },
            {
                path: '/whodrug-coding',
                key: '/whodrug-coding',
                redirect: '/coding/whodrug',
                hideInMenu: true,
            },
        ],
    },
    {
        path: '/forms-verification',
        key: '/forms-verification',
        label: 'Forms & Verification',
        icon: <FormOutlined />,
        element: FormsVerification,
    },
    {
        path: '/cra-activity',
        key: '/cra-activity',
        label: 'CRA Activity',
        icon: <TeamOutlined />,
        element: CRAActivity,
    },
    {
        path: '/reports',
        key: '/reports',
        label: 'Study Reports',
        icon: <FileTextOutlined />,
        element: Reports,
    },
    {
        path: '/studies',
        key: '/studies',
        label: 'Studies',
        icon: <DatabaseOutlined />,
        element: Studies,
    },
    {
        path: '/studies/:studyId',
        key: '/studies/:studyId',
        element: Studies,
        hideInMenu: true,
    },
    {
        path: '/alerts',
        key: '/alerts',
        label: 'Alerts',
        icon: <AlertOutlined />,
        element: Alerts,
    },
    {
        path: '/agents',
        key: '/agents',
        label: 'AI Agents',
        icon: <RobotOutlined />,
        element: Agents,
    },
    {
        path: '/conversational',
        key: '/conversational',
        label: 'AI Assistant',
        icon: <MessageOutlined />,
        element: Conversational,
    },
]
