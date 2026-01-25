# Neural Clinical Data Mesh Web Application

A real-time web application for clinical trial data monitoring with AI-powered insights and live patient cleanliness tracking.

## Features

- **Real-Time Monitoring**: Live patient status updates with WebSocket connections
- **AI-Powered Insights**: LongCat AI integration for enhanced reasoning and recommendations
- **Interactive Dashboards**: Live cleanliness charts and operational velocity metrics
- **Agent Swarm**: Rex, Codex, and Lia agents working together for comprehensive analysis
- **Patient Drill-Down**: Detailed patient views with blocking factor analysis
- **Alert System**: Real-time notifications for patient status changes

## Architecture

The application consists of:

1. **RealTimeDataMonitor**: Core monitoring system with WebSocket server
2. **LiveCleanlinessEngine**: Dynamic rule-based patient status evaluation
3. **Agent Framework**: AI agents enhanced with LongCat reasoning
4. **Dashboard Visualizer**: Real-time charts and metrics
5. **Web Application**: Flask-based UI with SocketIO for live updates

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements-web.txt
```

2. Set up environment variables:
```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env with your LongCat API credentials
```

3. Ensure clinical data is available in the expected directory structure.

## Usage

### Running the Flask Application

```bash
python web_app.py --framework flask --host 0.0.0.0 --port 5000
```

Navigate to `http://localhost:5000` in your browser.

### Running the Dash Dashboard

```bash
python web_app.py --framework dash --host 0.0.0.0 --port 8050
```

Navigate to `http://localhost:8050` in your browser.

### Command Line Options

- `--data-path`: Path to clinical data files (default: ../QC Anonymized Study Files)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 5000 for Flask, 8050 for Dash)
- `--framework`: Web framework to use ('flask' or 'dash', default: flask)
- `--debug`: Enable debug mode

## API Endpoints

### Flask Application

- `GET /`: Main dashboard page
- `GET /api/dashboard-data`: Current dashboard data (JSON)
- `GET /api/patient/<subject_id>`: Detailed patient information
- `GET /api/agent-insights`: Current agent insights and recommendations

### WebSocket Events

- `connect`: Client connection established
- `disconnect`: Client disconnected
- `patient_status_change`: Real-time patient status updates
- `dashboard_update`: Dashboard data refresh notifications

## Data Flow

1. **Data Ingestion**: Clinical data loaded from QC Anonymized Study Files
2. **Patient Twin Building**: Digital twins created for each patient
3. **Live Monitoring**: Continuous evaluation of patient cleanliness
4. **AI Enhancement**: LongCat AI provides reasoning for status changes
5. **Real-Time Updates**: WebSocket broadcasts updates to connected clients
6. **Dashboard Rendering**: Live charts and metrics updated in real-time

## Configuration

### Environment Variables

- `LONGCAT_API_KEY`: LongCat AI API key
- `LONGCAT_BASE_URL`: LongCat API base URL (optional)
- `OPENAI_API_KEY`: OpenAI API key (fallback)
- `ANTHROPIC_API_KEY`: Anthropic API key (fallback)

### Monitoring Settings

- Check interval: 60 seconds (configurable in RealTimeDataMonitor)
- WebSocket port: 8765 (configurable)
- Alert thresholds: Configurable in LiveCleanlinessEngine

## Development

### Project Structure

```
clinical_dataflow_optimizer/
├── web_app.py                 # Main web application
├── templates/
│   └── index.html            # Flask template
├── core/
│   ├── real_time_monitor.py  # Real-time monitoring system
│   └── longcat_integration.py # AI integration
├── agents/
│   └── agent_framework.py    # AI agent framework
├── visualization/
│   └── dashboard.py          # Dashboard components
└── config/
    └── settings.py           # Configuration management
```

### Testing

Run the web application tests:

```bash
python -m pytest test_web_app.py -v
```

### Extending the Application

- Add new dashboard components in `visualization/dashboard.py`
- Implement new agent types in `agents/agent_framework.py`
- Add new API endpoints in `web_app.py`
- Customize the UI in `templates/index.html`

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**: Check firewall settings and port availability
2. **Data Not Loading**: Verify data path and file permissions
3. **AI Features Not Working**: Check API keys in .env file
4. **Charts Not Updating**: Ensure WebSocket server is running

### Logs

Check the console output for detailed error messages and monitoring status.

## License

This project is part of the Clinical Dataflow Optimizer system.