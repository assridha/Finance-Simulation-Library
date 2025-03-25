import React from 'react';
import { ThemeProvider, createTheme, CssBaseline, Container, Typography } from '@mui/material';
import StockSimulator from './components/StockSimulator';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Typography variant="h3" component="h1" sx={{ my: 4, textAlign: 'center' }}>
          Stock Price Simulator
        </Typography>
        <StockSimulator />
      </Container>
    </ThemeProvider>
  );
}

export default App;
