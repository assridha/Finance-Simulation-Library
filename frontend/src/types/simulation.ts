export interface SimulationResult {
    current_price: number;
    simulated_paths: number[][];
    statistics: {
        expected_return: number;
        max: number;
        mean: number;
        min: number;
        prob_above_current: number;
        std: number;
        volatility: number;
    };
    symbol: string;
    timestamp: string;
    timestamps: string[];
}

export interface SimulationRequest {
    symbol: string;
    days_to_simulate: number;
    num_simulations: number;
} 