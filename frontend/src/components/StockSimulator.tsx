import React, { useState, useEffect, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Box, Button, TextField, Typography, Paper, Grid } from '@mui/material';
import axios from 'axios';
import { SimulationResult, SimulationRequest } from '../types/simulation';
import { Data } from 'plotly.js';
import { debounce } from 'lodash';

interface HistoricalData {
    timestamps: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
    symbol: string;
    period: string;
    interval: string;
}

const StockSimulator: React.FC = () => {
    const [symbol, setSymbol] = useState<string>('');
    const [daysToSimulate, setDaysToSimulate] = useState<number>(30);
    const [numSimulations, setNumSimulations] = useState<number>(5);
    const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
    const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');
    const [lastFetchedSymbol, setLastFetchedSymbol] = useState<string>('');

    // Memoize the API base URL
    const API_BASE = 'http://localhost:5001';

    const fetchHistoricalData = useCallback(async (symbolToFetch: string) => {
        // Don't fetch if we already have data for this symbol
        if (symbolToFetch === lastFetchedSymbol && historicalData) {
            return historicalData;
        }

        if (!symbolToFetch) return null;
        
        try {
            const response = await axios.get(`${API_BASE}/api/market-data/historical?symbol=${symbolToFetch}&period=6mo&interval=1d`);
            setHistoricalData(response.data);
            setLastFetchedSymbol(symbolToFetch);
            return response.data;
        } catch (err) {
            console.error('Failed to fetch historical data:', err);
            setError('Failed to fetch historical data');
            return null;
        }
    }, [lastFetchedSymbol, historicalData]); // Add proper dependencies

    // Memoize simulation function with proper dependencies
    const runSimulation = useCallback(async (request: SimulationRequest) => {
        try {
            const response = await axios.post(`${API_BASE}/api/market/simulate`, request);
            setSimulationResult(response.data);
            setLoading(false);
        } catch (err) {
            setError('Failed to run simulation. Please check the symbol and try again.');
            console.error(err);
            setLoading(false);
        }
    }, []);

    // Debounce the simulation to prevent rapid API calls
    const debouncedSimulate = useCallback(
        debounce((request: SimulationRequest) => runSimulation(request), 1000),
        [runSimulation]
    );

    const handleSimulate = async () => {
        if (loading || !symbol) return;
        
        try {
            setLoading(true);
            setError('');
            
            // Only fetch historical data if we don't have it for this symbol
            if (symbol !== lastFetchedSymbol) {
                const histData = await fetchHistoricalData(symbol);
                if (!histData) {
                    setLoading(false);
                    return;
                }
            }

            const request: SimulationRequest = {
                symbol: symbol.toUpperCase(),
                days_to_simulate: daysToSimulate,
                num_simulations: numSimulations
            };

            debouncedSimulate(request);
        } catch (err) {
            setError('Failed to run simulation. Please check the symbol and try again.');
            console.error(err);
            setLoading(false);
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            debouncedSimulate.cancel();
        };
    }, [debouncedSimulate]);

    const renderPlot = () => {
        if (!simulationResult) return null;

        const traces: Data[] = [];

        // Add historical candlestick data if available
        if (historicalData) {
            traces.push({
                x: historicalData.timestamps,
                open: historicalData.open,
                high: historicalData.high,
                low: historicalData.low,
                close: historicalData.close,
                type: 'candlestick' as const,
                name: 'Historical',
                showlegend: true,
                increasing: { line: { color: '#00C805' } },
                decreasing: { line: { color: '#FF3319' } },
                xaxis: 'x',
                yaxis: 'y'
            });
        }

        // Add simulation paths with a single legend entry
        let firstSimulationAdded = false;
        simulationResult.simulated_paths.forEach((path, index) => {
            traces.push({
                x: simulationResult.timestamps,
                y: path,
                type: 'scatter' as const,
                mode: 'lines' as const,
                name: firstSimulationAdded ? '' : 'Simulations',
                showlegend: !firstSimulationAdded,
                opacity: 0.3,
                line: { color: '#2196F3' },
                legendgroup: 'simulations'
            });
            firstSimulationAdded = true;
        });

        // Add mean line
        const meanValues = simulationResult.simulated_paths.reduce((acc, path) => {
            path.forEach((value, i) => {
                acc[i] = (acc[i] || 0) + value / simulationResult.simulated_paths.length;
            });
            return acc;
        }, [] as number[]);

        traces.push({
            x: simulationResult.timestamps,
            y: meanValues,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'Mean Prediction',
            line: { color: '#FFA726', width: 2 }
        });

        return (
            <Plot
                data={traces}
                layout={{
                    title: {
                        text: `Monte Carlo Simulation - ${simulationResult.symbol}`,
                        font: {
                            size: 24,
                            color: '#fff'
                        }
                    },
                    xaxis: { 
                        title: {
                            text: 'Date',
                            font: {
                                size: 14,
                                color: '#fff'
                            }
                        },
                        rangeslider: { visible: false },
                        type: 'date',
                        showgrid: true,
                        gridcolor: 'rgba(128, 128, 128, 0.2)',
                        tickfont: { size: 12, color: '#fff' },
                        tickformat: '%Y-%m-%d',
                        zeroline: false
                    },
                    yaxis: { 
                        title: {
                            text: 'Price ($)',
                            font: {
                                size: 14,
                                color: '#fff'
                            }
                        },
                        autorange: true,
                        showgrid: true,
                        gridcolor: 'rgba(128, 128, 128, 0.2)',
                        tickfont: { size: 12, color: '#fff' },
                        tickprefix: '$',
                        zeroline: false
                    },
                    showlegend: true,
                    height: 600,
                    legend: {
                        x: 0,
                        y: 1,
                        traceorder: 'normal',
                        orientation: 'h',
                        bgcolor: 'rgba(255, 255, 255, 0.1)',
                        font: { 
                            size: 12,
                            color: '#fff'
                        }
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 50, b: 50, l: 50, r: 50 },
                    font: {
                        family: 'Arial, sans-serif',
                        size: 12,
                        color: '#fff'
                    }
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
            />
        );
    };

    return (
        <Box sx={{ p: 3 }}>
            <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            label="Stock Symbol"
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                            placeholder="e.g., AAPL"
                            disabled={loading}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            type="number"
                            label="Days to Simulate"
                            value={daysToSimulate}
                            onChange={(e) => setDaysToSimulate(Number(e.target.value))}
                            inputProps={{ min: 1, max: 365 }}
                            disabled={loading}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <TextField
                            fullWidth
                            type="number"
                            label="Number of Simulations"
                            value={numSimulations}
                            onChange={(e) => setNumSimulations(Number(e.target.value))}
                            inputProps={{ min: 1, max: 1000 }}
                            disabled={loading}
                        />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                        <Button
                            fullWidth
                            variant="contained"
                            onClick={handleSimulate}
                            disabled={loading || !symbol}
                        >
                            {loading ? 'Running...' : 'Run Simulation'}
                        </Button>
                    </Grid>
                </Grid>
            </Paper>

            {error && (
                <Typography color="error" sx={{ mb: 2 }}>
                    {error}
                </Typography>
            )}

            {loading && !simulationResult && (
                <Typography sx={{ mb: 2 }}>
                    Loading simulation data...
                </Typography>
            )}

            {simulationResult && (
                <Paper elevation={3} sx={{ p: 3 }}>
                    <Grid container spacing={2} sx={{ mb: 2 }}>
                        <Grid item xs={12} sm={6} md={3}>
                            <Typography variant="subtitle2">Current Price</Typography>
                            <Typography variant="h6">${simulationResult?.current_price?.toFixed(2) || 'N/A'}</Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <Typography variant="subtitle2">Expected Return</Typography>
                            <Typography variant="h6">{(simulationResult?.statistics?.expected_return * 100)?.toFixed(2) || 'N/A'}%</Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <Typography variant="subtitle2">Probability Above Current</Typography>
                            <Typography variant="h6">{(simulationResult?.statistics?.prob_above_current * 100)?.toFixed(2) || 'N/A'}%</Typography>
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <Typography variant="subtitle2">30-Day Volatility</Typography>
                            <Typography variant="h6">{simulationResult?.statistics?.volatility?.toFixed(2) || 'N/A'}%</Typography>
                        </Grid>
                    </Grid>
                    {renderPlot()}
                </Paper>
            )}
        </Box>
    );
};

export default StockSimulator; 