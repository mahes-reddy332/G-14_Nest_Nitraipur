# Frontend Design Specification for Clinical Trial Operational Dataflow Metrics Web Application

## Application Overview
Design a responsive web application that provides real-time operational dataflow metrics for clinical trials. The application integrates data from multiple clinical trial systems to generate actionable insights, detect operational bottlenecks, and support scientific decision-making for Data Quality Teams (DQT), Clinical Research Associates (CRAs), and Investigational Sites.

---

## Core Data Dashboard Requirements

### 1. Main Navigation Structure

The application must include a hierarchical navigation system with the following primary sections:

#### Primary Navigation Tabs:
- **Dashboard Home** - Overview and summary metrics
- **Patient & Site Metrics** - EDC metrics and subject-level data
- **Data Quality** - Query metrics and data quality index
- **Visit Management** - Visit projection and missing visits tracking
- **Laboratory Data** - Lab reconciliation and missing ranges
- **Safety Monitoring** - SAE dashboard and adverse events
- **Coding & Reconciliation** - MedDRA and WHO Drug coding status
- **Forms & Verification** - SDV status, frozen/locked/signed forms
- **CRA Activity** - Monitoring visits and follow-ups
- **Reports & Analytics** - Derived metrics and clean data milestones

---

## 2. Dashboard Home Page

### Overview Section
Display high-level KPIs in card format:
- **Total Sites** (with active/inactive breakdown)
- **Total Subjects** (with enrollment status)
- **Clean Patient Percentage** (derived metric)
- **Data Quality Index Score** (weighted aggregate metric)
- **Open Queries Count** (with severity breakdown)
- **Missing Visits Count**
- **Overdue CRFs Count**
- **Unresolved SAEs Count**

### Visual Components Required:
- **Regional Performance Map**: Interactive world map or regional visualization showing:
  - Color-coded regions by data quality index
  - Bubble size representing number of sites
  - Click-to-drill-down functionality to country/site level

- **Trend Charts**:
  - Query resolution rate over time (line chart)
  - Enrollment progress vs. target (bar chart)
  - Data completeness percentage trend (area chart)

- **Alert Panel**:
  - Critical alerts highlighted in red
  - Warning alerts in amber
  - Information alerts in blue
  - Each alert must show: type, site/subject, timestamp, and action button

- **Quick Action Buttons**:
  - "Sites Requiring Immediate Attention"
  - "Generate CRA Report"
  - "Check Data Readiness for Analysis"
  - "View All Open Issues"

---

## 3. Patient & Site Metrics Dashboard (CPID_EDC_Metrics)

### Filter Panel (Left Sidebar or Top Bar)
Multi-level filtering capability:
- **Region** (dropdown, multi-select)
- **Country** (dropdown, multi-select, cascading from Region)
- **Site** (dropdown with search, multi-select)
- **Subject ID** (search field with autocomplete)
- **Subject Status** (checkbox filters: Screened, Enrolled, Completed, Withdrawn, Screen Failed)
- **Date Range** (calendar date picker)
- **Data Quality Status** (Clean/Not Clean toggle)

### Subject Level Metrics Table
Display comprehensive subject-level data with following columns:
- Region
- Country
- Site ID/Name
- Subject ID
- Subject Status
- Enrollment Date
- Last Visit Date
- Total Visits Planned vs. Completed
- Missing Visits Count (with % calculation)
- Missing Pages Count (with % calculation)
- Open Queries (total count, by type: Data Query, Protocol Deviation, Safety Query)
- Non-Conformant Data Count
- SDV Status (percentage verified)
- Frozen Forms Count
- Locked Forms Count
- Signed Forms Count
- Overdue CRFs Count
- Inactivated Folders Count
- Clean Patient Status (Yes/No indicator with color coding)
- Last Update Timestamp

### Table Features Required:
- **Sortable columns** (click header to sort ascending/descending)
- **Column visibility toggle** (allow users to show/hide columns)
- **Export functionality** (Excel, CSV, PDF)
- **Row selection** (checkboxes for bulk actions)
- **Drill-down capability** (click row to view detailed subject profile)
- **Conditional formatting**:
  - Red highlight for critical issues (>10 open queries, >20% missing data)
  - Yellow highlight for warnings (5-10 open queries, 10-20% missing data)
  - Green highlight for clean patients
- **Pagination** (with options for 25/50/100/All rows per page)
- **Search within table** (real-time filtering)

### Derived Metrics Visualization Panel
Display calculated percentages in gauge charts or progress bars:
- % Missing Visits (by site, by region)
- % Missing Pages (by visit type)
- % Clean CRFs (overall and by site)
- % Open Queries Resolved (with time-based targets)
- % Non-Conformant Data
- % Verification Status Complete

---

## 4. Visit Management Section

### Visit Projection Tracker Interface

#### Missing Visits Table (Primary View)
Display all projected visits not yet occurred or entered:
- Subject ID
- Site ID/Name
- Visit Name/Number
- Projected Visit Date
- Days Overdue (calculated, color-coded: >30 days = red, 15-30 = yellow, <15 = orange)
- Visit Type (Screening, Baseline, Follow-up, End of Study)
- Last Contact Date
- CRA Assigned
- Follow-up Status (Pending/In Progress/Contacted/Resolved)
- Action Buttons (Send Reminder, Mark Completed, Add Comment)

#### Visual Analytics for Visits:
- **Calendar Heatmap**: Show overdue visits by day/week
- **Site Comparison Chart**: Bar chart showing missing visits count per site
- **Visit Type Breakdown**: Pie chart showing which visit types are most commonly missed
- **Trend Analysis**: Line graph showing overdue visits trend over past 3/6/12 months

#### Compliance Metrics Panel:
- Average Days Overdue (by site, by region)
- Visit Compliance Rate (percentage of visits completed on time)
- Sites with >5 Overdue Visits (list with counts)
- Subjects with >3 Overdue Visits (list with counts)

---

## 5. Laboratory Data Management Section

### Missing Lab Data Interface

Display all instances where lab data is incomplete:

#### Missing Lab Names & Ranges Table:
- Subject ID
- Site ID/Name
- Visit Name
- Lab Test Name
- Missing Element (Lab Name/Reference Range/Unit)
- Collection Date
- Received Date
- Days Since Collection
- Priority Level (Critical/High/Medium/Low based on test type)
- Assigned To (data manager/CRA)
- Resolution Status (Open/In Progress/Resolved)
- Comments/Notes field

#### Lab Reconciliation Dashboard:
- **Summary Cards**:
  - Total Missing Lab Names
  - Total Missing Reference Ranges
  - Total Missing Units
  - Average Resolution Time
  
- **Visual Elements**:
  - Lab type breakdown (pie chart: Hematology, Chemistry, Urinalysis, etc.)
  - Resolution timeline (Gantt chart or timeline view)
  - Site performance on lab data quality (horizontal bar chart)

#### Issue Tracking Panel:
- Open lab reconciliation issues count
- Categorized by severity
- Quick filters: By Test Type, By Site, By Age of Issue

---

## 6. Safety Monitoring - SAE Dashboard

### Two-Tab Interface Required:

#### Tab 1: SAE Dashboard - Data Management View
Display all SAE discrepancies from data management perspective:
- Subject ID
- Site ID/Name
- SAE Description/Term
- Onset Date
- Report Date
- Discrepancy Type
- Discrepancy Status (Open/Under Review/Resolved)
- Days Open
- Assigned Data Manager
- Last Update Date
- Priority (Expedited/Standard)
- Action Required field
- Comment thread

#### Tab 2: SAE Dashboard - Safety View
Display SAE review status from safety team perspective:
- Subject ID
- Site ID/Name
- SAE Term
- Onset Date
- Severity (Mild/Moderate/Severe)
- Causality Assessment
- Expectedness
- Review Status (Pending Initial Review/Under Review/Completed)
- Medical Review Date
- Safety Physician Assigned
- Follow-up Required (Yes/No)
- Regulatory Reporting Status
- Comment thread

#### SAE Visual Analytics:
- **SAE Trend Chart**: Timeline showing SAE reports over study duration
- **Severity Distribution**: Pie chart (Mild/Moderate/Severe)
- **Site Comparison**: Bar chart showing SAE count per site
- **Resolution Metrics**:
  - Average time from report to resolution
  - Percentage of SAEs reviewed within target timeframe
  - Outstanding safety reviews count

#### Alert System for SAEs:
- Critical alerts for new SAEs requiring expedited review
- Notifications for SAEs approaching review deadlines
- Escalation indicators for overdue safety reviews

---

## 7. Data Quality & Query Management Section

### Query Metrics Dashboard

#### Query Summary Panel:
Display aggregate query statistics:
- **Total Open Queries** (with trend indicator)
- **Queries by Type**:
  - Data Queries
  - Protocol Deviations
  - Safety Queries
  - Lab Queries
  - Other
- **Query Age Distribution**:
  - <7 days
  - 7-14 days
  - 15-30 days
  - >30 days (critical, highlighted in red)
- **Average Resolution Time** (by query type)

#### Query Details Table:
- Query ID
- Subject ID
- Site ID/Name
- Visit/Form Name
- Query Type
- Query Field
- Query Text/Description
- Opened Date
- Days Open
- Query Status (Open/Answered/Closed/Cancelled)
- Assigned To (CRA/Site)
- Response Due Date
- Last Response Date
- Priority Level
- Action Buttons (View Details, Send Reminder, Close Query)

#### Query Resolution Tracking:
- **Resolution Rate Chart**: Percentage of queries resolved by week/month
- **Site Performance Matrix**: Heat map showing query resolution rates by site
- **CRA Performance**: Table showing queries resolved per CRA with avg resolution time
- **Query Aging Report**: Stacked bar chart showing age distribution of open queries over time

#### Non-Conformant Data Section:
- Total count of non-conformant data points
- Breakdown by data validation rule violated
- Sites with highest non-conformance rates
- Trend over time
- Export capability for detailed review

---

## 8. Coding & Reconciliation Dashboards

### Two Separate Sub-sections Required:

#### 8A. MedDRA Coding Dashboard (GlobalCodingReport_MedDRA)

Display all medical terms requiring MedDRA coding:

**Coding Status Table:**
- Subject ID
- Site ID/Name
- Term Type (Adverse Event/Medical History/Indication)
- Verbatim Term
- MedDRA Coded Term (if coded)
- Preferred Term (PT)
- High Level Term (HLT)
- System Organ Class (SOC)
- Coding Status (Uncoded/Pending Review/Coded/Approved)
- Coder Assigned
- Date Term Entered
- Date Coded
- Days Pending Coding
- Comments/Query field

**Coding Metrics Panel:**
- Total Terms Requiring Coding
- Uncoded Terms Count
- Coded but Pending Review Count
- Fully Approved Count
- Average Coding Turnaround Time
- Coding accuracy rate

**Visual Elements:**
- SOC Distribution Chart (bar chart showing most common SOCs)
- Coding Progress Timeline
- Site-wise Uncoded Terms (horizontal bar chart)

#### 8B. WHO Drug Coding Dashboard (GlobalCodingReport_WHODRA)

Display all medications requiring WHO Drug dictionary coding:

**Medication Coding Table:**
- Subject ID
- Site ID/Name
- Medication Type (Concomitant/Prior/Protocol)
- Verbatim Drug Name
- WHO Drug Coded Term (if coded)
- Drug Code
- ATC Classification
- Coding Status (Uncoded/Pending Review/Coded/Approved)
- Coder Assigned
- Date Medication Entered
- Date Coded
- Days Pending Coding
- Comments/Query field

**Medication Coding Metrics:**
- Total Medications Requiring Coding
- Uncoded Medications Count
- Coded but Pending Review Count
- Fully Approved Count
- Average Coding Turnaround Time

**Visual Elements:**
- ATC Class Distribution Chart
- Coding Progress over Time
- Site-wise Uncoded Medications

---

## 9. Forms & Verification Status Section

### Source Data Verification (SDV) Dashboard

#### SDV Status Overview:
Display verification status across all subjects and sites:

**SDV Summary Cards:**
- Total Forms Requiring SDV
- Forms 100% Verified
- Forms Partially Verified
- Forms Not Verified
- Average SDV Completion Rate
- Sites with <50% SDV Completion (alert list)

**SDV Details Table:**
- Subject ID
- Site ID/Name
- Visit Name
- Form Name
- Total Fields
- Fields Verified Count
- % Verified (progress bar visualization)
- SDV Status (Not Started/In Progress/Complete)
- CRA Performing SDV
- Last Monitoring Visit Date
- Next Planned Monitoring Visit
- Days Since Last SDV Activity

#### Form Status Tracking:

**Form Status Table:**
- Subject ID
- Site ID/Name
- Visit Name
- Form Name
- Form Status:
  - Frozen (count and list)
  - Locked (count and list)
  - Signed (count and list)
- Date Status Changed
- User Who Changed Status
- Signature Applied By (if signed)
- Signature Date

**Overdue CRFs Panel:**
- Subject ID
- Site ID/Name
- Form Name
- Expected Completion Date
- Days Overdue
- Priority (based on visit criticality)
- Assigned CRA
- Follow-up Status
- Action buttons (Send Reminder, Mark Exception, Add Note)

**Inactivated Forms & Log Lines:**
Display all deactivated data:
- Subject ID
- Site ID/Name
- Form/Log Line Name
- Inactivation Date
- Inactivated By
- Reason for Inactivation (dropdown: Protocol Deviation, Data Entry Error, Visit Not Conducted, Other)
- Detailed Explanation field
- Approval Status
- Approver Name
- Approval Date

---

## 10. CRA Activity Section

### CRA Monitoring Dashboard

#### CRA Performance Metrics:
Display individual and aggregate CRA activity:

**CRA Summary Cards:**
- Total Active CRAs
- Total Monitoring Visits Conducted (this month/quarter/year)
- Average Visits per CRA
- Sites per CRA (workload distribution)
- Average Time per Site Visit

**Monitoring Visits Log Table:**
- CRA Name
- Site ID/Name
- Visit Date
- Visit Type (Initiation/Routine/Close-out/For Cause)
- Findings Count
- Critical Findings Count
- Follow-up Items Count
- Visit Report Status (Draft/Submitted/Approved)
- Report Submission Date
- Next Planned Visit Date

**Follow-up Activity Tracker:**
- CRA Name
- Site ID/Name
- Follow-up Item Description
- Item Priority (Critical/High/Medium/Low)
- Date Identified
- Due Date
- Status (Open/In Progress/Resolved/Overdue)
- Days Since Identified
- Last Update Date
- Resolution Notes

#### CRA Workload Visualization:
- **Site Distribution Map**: Show which CRA is assigned to which sites geographically
- **Workload Balance Chart**: Bar chart showing number of sites per CRA
- **Visit Frequency Timeline**: Calendar view showing scheduled and completed visits
- **Performance Scorecard**: Compare CRAs on metrics like:
  - Query resolution rate for their sites
  - SDV completion rate
  - Average time to close findings
  - Site data quality scores

---

## 11. Reports & Analytics Section

### Clean Data Milestones Dashboard

#### Readiness Check Panel:
Display data readiness for statistical analysis or submission:

**Overall Readiness Indicators:**
- **Data Lock Readiness Score** (0-100%)
  - Based on: query closure, SDV completion, missing data resolution, coding completion
- **Planned vs Actual Cut-off Date** comparison
- **Blocking Issues Count** (critical items preventing data lock)
- **Days to Target Lock Date** (countdown timer)

**Milestone Tracking Table:**
- Milestone Name (Interim Analysis 1, Final Data Lock, etc.)
- Planned Date
- Forecasted Date (based on current progress)
- Status (On Track/At Risk/Delayed)
- Completion Percentage
- Blocking Issues List
- Owner/Responsible Person
- Last Updated

**Data Quality Index (DQI) Detailed View:**

Display weighted aggregate score calculation:
- **Parameter Weights Table**:
  - Missing Visits (weight %)
  - Missing Pages (weight %)
  - Open Queries (weight %)
  - Non-Conformant Data (weight %)
  - Unverified Forms (weight %)
  - Uncoded Terms (weight %)
  - Unresolved SAE Discrepancies (weight % - highest critical factor)
  
- **DQI Score by**:
  - Overall Study
  - By Region
  - By Country
  - By Site
  - By Subject

**Visual Representation:**
- DQI Score Gauge (0-100 scale with color zones: 0-60 Red, 60-80 Yellow, 80-100 Green)
- Heat map showing DQI scores across all sites
- Trend line showing DQI improvement over time
- Comparison chart: Current DQI vs. Target DQI

#### Derived Metrics Summary Report:

Display all calculated percentages:
- % Missing Visits (overall, by site, by region)
- % Missing Pages (overall, by visit type, by site)
- % Clean CRFs (by site, by visit type)
- % Open Queries (by type, by site)
- % Non-Conformant Data (by validation rule, by site)
- % Verification Status (SDV completion by site)
- % Clean Patients (patients with zero issues across all parameters)

**Export and Automation:**
- One-click report generation for CRA reports
- Automated summary of site performance
- Scheduled email reports option
- Custom report builder (select metrics, filters, format)

---

## 12. Third-Party Data Reconciliation (Compiled_EDRR)

### External Data Review Dashboard

#### Unresolved Issues Summary:
Display total unresolved data issues from third-party sources:

**Issue Summary Table:**
- Subject ID
- Site ID/Name
- Data Source (Lab, ECG, Imaging, etc.)
- Issue Type
- Issue Description
- Priority Level
- Date Identified
- Days Open
- Assigned To
- Status (Open/Under Review/Pending Response/Resolved)
- Expected Resolution Date
- Comments thread

**Prioritization Panel:**
- Critical Issues Requiring Immediate Action (top of list, red highlight)
- High Priority Issues (sorted by subject enrollment status)
- Medium and Low Priority Issues
- Total count by priority level

**Visual Metrics:**
- Issues by data source (pie chart)
- Resolution trend over time (line chart)
- Average time to resolution by issue type
- Site comparison for third-party data issues

---

## 13. Collaboration & Communication Features

### Built-in Collaboration Tools (Integrated Throughout Application):

#### Alert System:
- **Alert Creation**: Users can create custom alerts for specific conditions
- **Alert Types**:
  - Email notifications
  - In-app notifications (bell icon with badge count)
  - SMS alerts for critical issues (optional)
- **Alert Triggers**:
  - New critical query opened
  - SAE reported
  - Data quality threshold breach
  - Milestone deadline approaching
  - Site performance below threshold

#### Tagging System:
- Ability to tag team members in comments (@mention functionality)
- Tag by role (DQT, CRA, Site Coordinator, Medical Monitor)
- Tag subjects, sites, or specific data points for follow-up
- Tag categories for issues (Data Quality, Protocol Compliance, Safety, etc.)

#### Comments & Annotations:
- Comment threads on every table row (subject, query, visit, etc.)
- File attachment capability in comments
- Comment history with timestamps and user names
- Ability to mark comments as resolved
- Search within comments

#### Task Assignment:
- Assign tasks directly from any dashboard view
- Task status tracking (To Do/In Progress/Completed)
- Due date assignment with reminders
- Task owner assignment with workload visibility
- Task dependencies and relationships

---

## 14. AI-Powered Features Integration

### Natural Language Query Interface:

**Query Input Box** (prominently placed at top of dashboard):
- Large search bar with placeholder text: "Ask anything about your clinical trial data..."
- Examples shown below: 
  - "Which sites have the most missing visits?"
  - "Show me patients with unresolved safety queries"
  - "What is the data quality status for Site 101?"
  - "Which CRAs are underperforming?"

**AI Response Panel:**
- Display natural language results
- Show supporting data tables/charts
- Provide drill-down links to detailed views
- Offer follow-up question suggestions

### Generative AI Features:

**Automated Report Generation:**
- Button: "Generate CRA Report" on CRA Activity page
- Button: "Generate Site Performance Summary" on Patient & Site Metrics page
- Generated reports should include:
  - Executive summary paragraph
  - Key findings bullets
  - Visual charts
  - Recommended actions
  - Export to Word/PDF functionality

**Performance Summarization:**
- AI-generated summary of site performance (1-2 paragraph overview)
- Highlight positive trends and areas of concern
- Compare to study benchmarks

### Agentic AI Features:

**Risk-Based Recommendations Panel:**
- Display on Dashboard Home and relevant section pages
- Show AI-recommended actions based on current data signals:
  - "Recommend immediate monitoring visit to Site 105 due to 15 overdue visits and 8 critical queries"
  - "Suggest data quality review for Region APAC - DQI score dropped 12% this month"
  - "Flag Subject 12345 for safety follow-up - 3 open SAE discrepancies >14 days old"

**Automated Issue Prioritization:**
- AI ranks issues by impact and urgency
- Color-coded priority assignments
- Suggested action plans with estimated resolution time

**Predictive Analytics:**
- Forecast data lock readiness date based on current trends
- Predict sites likely to have data quality issues
- Estimate query resolution time based on historical patterns
- Alert on trending problems before they become critical

---

## 15. Technical UI/UX Requirements

### Responsive Design:
- Must work seamlessly on desktop (1920x1080 and above)
- Tablet-optimized views (iPad, Surface)
- Mobile-responsive for monitoring on-the-go (key metrics only)

### Performance Requirements:
- Page load time <2 seconds for all dashboard views
- Real-time data updates (auto-refresh every 5 minutes, with manual refresh button)
- Smooth scrolling for large tables (virtualized rendering for 1000+ rows)
- Lazy loading for charts and detailed data

### Visual Design Standards:

#### Color Scheme:
- Primary brand colors for navigation and headers
- Status Colors (standardized across all views):
  - **Green**: Clean/Complete/On Track (#28a745)
  - **Yellow/Amber**: Warning/Needs Attention (#ffc107)
  - **Red**: Critical/Overdue/High Priority (#dc3545)
  - **Blue**: Information/In Progress (#007bff)
  - **Gray**: Inactive/Cancelled/Not Applicable (#6c757d)

#### Typography:
- Clear, professional sans-serif font (e.g., Roboto, Open Sans, Inter)
- Font sizes: 
  - Headers: 24-32px
  - Subheaders: 18-20px
  - Body text: 14-16px
  - Table data: 13-14px
- Adequate line spacing for readability

#### Layout Principles:
- Consistent spacing and padding (8px grid system)
- Clear visual hierarchy
- Adequate white space to avoid clutter
- Card-based layouts for metric summaries
- Sticky headers for tables and navigation

### Interactive Elements:

#### Tooltips:
- Hover tooltips for all abbreviations and metrics
- Explain calculation methods for derived metrics
- Show full text for truncated fields

#### Loading States:
- Skeleton screens or spinners during data loading
- Progress indicators for long-running operations
- "Loading..." text with estimated time if >3 seconds

#### Empty States:
- Friendly messages when no data exists
- Suggestions for next steps
- Visual illustrations (not just text)

#### Error Handling:
- Clear error messages with suggested resolution
- "Retry" buttons for failed operations
- Contact support option for persistent errors

### Accessibility Requirements:
- WCAG 2.1 Level AA compliance
- Keyboard navigation support (tab through all interactive elements)
- Screen reader compatible
- High contrast mode option
- Text size adjustment option
- Color-blind friendly palette (don't rely solely on color for information)

---

## 16. Data Export & Integration Features

### Export Capabilities (Available on All Tables and Charts):
- **Excel Export**: Full table with all columns, formatted
- **CSV Export**: Raw data for further analysis
- **PDF Export**: Print-friendly formatted report
- **PowerPoint Export**: Charts and summary for presentations
- **Email Report**: Send current view/report via email

### API Integration Points:
- Real-time data sync with clinical trial systems (EDC, CTMS, eTMF)
- Webhook support for external notifications
- RESTful API for custom integrations

### Scheduled Reports:
- Set up recurring reports (daily/weekly/monthly)
- Recipient list management
- Custom filters and views saved for scheduled reports
- Delivery time configuration

---

## 17. User Management & Permissions

### Role-Based Access Control:

#### User Roles and Dashboard Access:
- **Data Quality Manager**: Full access to all dashboards
- **Clinical Research Associate (CRA)**: Access to assigned sites, monitoring, query, and SDV sections
- **Site Coordinator**: Access to their site's data only, limited editing permissions
- **Medical Monitor**: Full access to safety, SAE, and AE sections
- **Statistician**: Read-only access to clean data milestone and derived metrics
- **Study Manager**: Executive dashboard view with all high-level metrics

### User Profile Section:
- User name and role display (top right corner)
- Profile settings access
- Notification preferences
- Time zone settings
- Language selection (if multi-lingual)
- Logout button

---

## 18. Search and Filter Capabilities

### Global Search (Top Navigation Bar):
- Universal search box to find:
  - Subjects by ID or other identifiers
  - Sites by ID or name
  - Queries by ID or text
  - Forms by name
  - CRAs by name
- Auto-suggest results as user types
- Recent searches saved

### Advanced Filtering (Every Dashboard Section):
- Multi-select dropdowns for categorical data
- Date range pickers with presets (Last 7 days, Last 30 days, This Quarter, etc.)
- Numeric range filters (e.g., Days Overdue: 1-30, 31-60, 60+)
- Boolean toggles (Yes/No, True/False)
- Save filter combinations as "Saved Views"
- Share saved views with team members
- Default views per user role

---

## 19. Dashboard Customization

### Widget Customization:
- Drag-and-drop dashboard widgets
- Resize charts and panels
- Show/hide sections based on user preference
- Set default landing page per user

### Personalization Features:
- Custom color themes (light/dark mode)
- Save frequently used filters and searches
- Bookmark specific views or subjects
- Custom metrics and KPI selection

---

## 20. Help & Documentation

### Integrated Help System:
- **Help Icon** (?) next to complex features
- **Contextual Help**: Explanations specific to current page
- **Video Tutorials**: Embedded short videos for key workflows
- **User Guide**: Comprehensive documentation accessible from help menu
- **FAQ Section**: Common questions and answers
- **Keyboard Shortcuts Guide**: Accessible via keyboard shortcut (Ctrl+/)

### Tooltips and Field Definitions:
- Hover definitions for all clinical trial terminology
- Explanation of derived metric calculations
- Links to protocol-specific documentation where applicable

---

## 21. Performance Monitoring & System Health

### System Status Indicator (Footer or Header):
- **Green Dot**: All systems operational
- **Yellow Dot**: Some services experiencing delays
- **Red Dot**: System issues, contact support
- Last data refresh timestamp
- Data source status (EDC connected, Lab system connected, etc.)

---

## Summary of Critical UI Elements Across All Pages:

### Mandatory Header Components:
- Application logo and name
- Primary navigation menu
- Global search bar
- AI natural language query box
- Notifications bell icon (with unread count badge)
- User profile and settings
- Help/documentation access

### Mandatory Sidebar/Filter Panel (Where Applicable):
- Hierarchical filters (Region > Country > Site > Subject)
- Date range selectors
- Status filters
- Save/Load filter presets
- Clear all filters button
- Apply filters button

### Mandatory Table Features (All Data Tables):
- Column sorting (ascending/descending)
- Column show/hide toggle
- Row selection checkboxes
- Inline actions (view details, edit, add comment)
- Pagination controls
- Rows per page selector
- Export buttons (Excel, CSV, PDF)
- Search within table

### Mandatory Visual Chart Features:
- Interactive legends (click to show/hide series)
- Zoom and pan capabilities
- Drill-down on click
- Export chart as image
- Data table view toggle (switch between chart and table)
- Tooltip showing exact values on hover

### Mandatory Footer Components:
- Copyright and version information
- Privacy policy and terms of service links
- Contact support link
- System status indicator
- Last updated timestamp

---

## Final Notes for Frontend Development:

1. **Data Refresh**: All dashboards must show a "Last Updated" timestamp and provide a manual refresh button. Auto-refresh should occur every 5 minutes for critical dashboards (Dashboard Home, SAE Dashboard, Query Metrics).

2. **Loading Performance**: Use lazy loading and virtualization for large datasets. Tables with >500 rows should use virtual scrolling.

3. **Consistent Metrics**: Ensure all derived metrics calculations (% missing visits, % clean CRFs, Data Quality Index) use the same methodology across all dashboard views.

4. **Mobile Considerations**: For mobile views, prioritize critical alerts, summary KPIs, and simplified tables. Full functionality remains on desktop/tablet.

5. **Real-time Collaboration**: Changes made by one user (e.g., adding a comment, closing a query) should be visible to other users within 30 seconds without requiring a page refresh.

6. **Clean Patient Definition**: Clearly display the criteria for "Clean Patient" status. A patient must have:
   - Zero missing visits
   - Zero unresolved queries
   - All required forms verified and signed
   - No non-conformant data
   - All coding completed
   - No open SAE discrepancies

7. **Data Quality Index Transparency**: Provide a detailed breakdown of how the DQI is calculated, including individual parameter scores and weights. Users should be able to customize weights based on study-specific priorities.

8. **Audit Trail**: For critical actions (form sign-off, query closure, SAE resolution), maintain and display an audit trail showing who made changes and when.

9. **Bulk Actions**: Allow bulk operations where applicable (e.g., bulk query closure, bulk assignment of tasks, bulk export of subjects).

10. **Offline Capability**: Consider offline mode for CRAs in the field—allow viewing of previously loaded data and sync changes when reconnected.

This comprehensive frontend specification covers all requirements from both the User Requirements Document and Dataset Guidance Document, ensuring a complete, user-friendly, and functionally rich clinical trial operational dataflow metrics web application.
