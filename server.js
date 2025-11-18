/**
 * Ultra-Advanced Trading Bridge Server
 * High-performance bridge between MT5 and AI prediction system
 */

const express = require("express");
const WebSocket = require("ws");
const Redis = require("redis");
const cors = require("cors");
const helmet = require("helmet");
const compression = require("compression");
const rateLimit = require("express-rate-limit");
const winston = require("winston");
const { performance } = require("perf_hooks");

// ===================== Configuration =====================
const config = {
  // Server settings
  HTTP_PORT: process.env.PORT || 3000,
  WS_PORT: process.env.WS_PORT || 8080,

  // Redis settings
  REDIS_HOST: process.env.REDIS_HOST || "localhost",
  REDIS_PORT: process.env.REDIS_PORT || 6379,

  // AI Worker settings
  AI_WORKER_WS: process.env.AI_WORKER_WS || "ws://localhost:8765",

  // Performance settings
  MAX_CONCURRENT_PREDICTIONS: 10,
  PREDICTION_TIMEOUT: 30000, // ms
  CACHE_TTL: 60, // seconds

  // Security settings
  RATE_LIMIT_WINDOW: 5000, // 5detik
  RATE_LIMIT_MAX: 100,

  // Data settings
  MIN_CANDLES: 50,
  MAX_CANDLES: 500,
  DEFAULT_TIMEFRAME: "M15",
};

// ===================== Logger Setup =====================
const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "combined.log" }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf((info) => {
          return `${info.level}: ${JSON.stringify(info, null, 2)}`;
        })
      ),
    }),
  ],
});

// ===================== Redis Client =====================
class RedisManager {
  constructor() {
    this.client = Redis.createClient({
      url: `redis://${config.REDIS_HOST}:${config.REDIS_PORT}`,
    });

    this.publisher = Redis.createClient({
      url: `redis://${config.REDIS_HOST}:${config.REDIS_PORT}`,
    });

    this.subscriber = Redis.createClient({
      url: `redis://${config.REDIS_HOST}:${config.REDIS_PORT}`,
    });

    this.connectAll();
  }

  async connectAll() {
    try {
      await this.client.connect();
      await this.publisher.connect();
      await this.subscriber.connect();

      console.log("Redis connected");
    } catch (err) {
      console.error("Redis connection failed:", err);
    }
  }

  async cacheData(key, data, ttl = config.CACHE_TTL) {
    await this.client.setEx(key, ttl, JSON.stringify(data));
  }

  async getCachedData(key) {
    const raw = await this.client.get(key);
    return raw ? JSON.parse(raw) : null;
  }

  async publish(channel, data) {
    await this.publisher.publish(channel, JSON.stringify(data));
  }

  subscribe(channel, callback) {
    this.subscriber.subscribe(channel, (message) => {
      callback(JSON.parse(message));
    });
  }
}

// ===================== AI Worker Connection =====================
class AIWorkerClient {
  constructor() {
    this.ws = null;
    this.connected = false;
    this.requestQueue = [];
    this.pendingRequests = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;

    this.connect();
  }

  connect() {
    try {
      this.ws = new WebSocket(config.AI_WORKER_WS);

      this.ws.on("open", () => {
        logger.info("Connected to AI Worker");
        this.connected = true;
        this.reconnectAttempts = 0;

        // Process queued requests
        this.processQueue();
      });

      this.ws.on("message", (data) => {
        try {
          const message = JSON.parse(data);
          this.handleWorkerMessage(message);
        } catch (err) {
          logger.error("Error parsing worker message:", err);
        }
      });

      this.ws.on("close", () => {
        logger.warn("Disconnected from AI Worker");
        this.connected = false;
        this.reconnect();
      });

      this.ws.on("error", (err) => {
        logger.error("AI Worker connection error:", err);
      });
    } catch (err) {
      logger.error("Failed to connect to AI Worker:", err);
      this.reconnect();
    }
  }

  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      logger.info(`Reconnecting to AI Worker in ${delay}ms...`);
      setTimeout(() => this.connect(), delay);
    } else {
      logger.error("Max reconnection attempts reached");
    }
  }

  async predict(data) {
    return new Promise((resolve, reject) => {
      const requestId = this.generateRequestId();
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error("Prediction timeout"));
      }, config.PREDICTION_TIMEOUT);

      this.pendingRequests.set(requestId, {
        resolve,
        reject,
        timeout,
        timestamp: Date.now(),
      });

      const request = {
        id: requestId,
        type: 'predict',
        data: {
            candles: data.candles,             // sinkron dengan Python
            timeframe: data.timeframe,
            confidence_threshold: data.confidence_threshold
        }
    };


      if (this.connected && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(request));
      } else {
        this.requestQueue.push(request);
      }
    });
  }

  handleWorkerMessage(message) {
    const requestId = message.id;

    if (requestId && this.pendingRequests.has(requestId)) {
      const pending = this.pendingRequests.get(requestId);
      clearTimeout(pending.timeout);
      this.pendingRequests.delete(requestId);

      if (message.error) {
        pending.reject(new Error(message.error));
      } else {
        pending.resolve(message);
      }
    }
  }

  processQueue() {
    while (this.requestQueue.length > 0 && this.connected) {
      const request = this.requestQueue.shift();
      this.ws.send(JSON.stringify(request));
    }
  }

  generateRequestId() {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getStatus() {
    return {
      connected: this.connected,
      pendingRequests: this.pendingRequests.size,
      queuedRequests: this.requestQueue.length,
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

// ===================== Market Data Manager =====================
class MarketDataManager {
  constructor() {
    this.dataBuffer = new Map();
    this.maxBufferSize = 1000;
    this.indicators = {};
  }

  addCandle(symbol, timeframe, candle) {
    const key = `${symbol}_${timeframe}`;

    if (!this.dataBuffer.has(key)) {
      this.dataBuffer.set(key, []);
    }

    const buffer = this.dataBuffer.get(key);
    buffer.push(candle);

    // Maintain buffer size
    if (buffer.length > this.maxBufferSize) {
      buffer.shift();
    }

    // Update indicators
    this.updateIndicators(symbol, timeframe);
  }

  getCandles(symbol, timeframe, count = 100) {
    const key = `${symbol}_${timeframe}`;
    const buffer = this.dataBuffer.get(key) || [];

    return buffer.slice(-count);
  }

  updateIndicators(symbol, timeframe) {
    const candles = this.getCandles(symbol, timeframe, 100);

    if (candles.length < 20) return;

    const key = `${symbol}_${timeframe}`;

    // Calculate basic indicators for quick reference
    const closes = candles.map((c) => c.close);
    const highs = candles.map((c) => c.high);
    const lows = candles.map((c) => c.low);
    const volumes = candles.map((c) => c.volume);

    this.indicators[key] = {
      lastPrice: closes[closes.length - 1],
      change24h: ((closes[closes.length - 1] - closes[0]) / closes[0]) * 100,
      highestHigh: Math.max(...highs),
      lowestLow: Math.min(...lows),
      avgVolume: volumes.reduce((a, b) => a + b, 0) / volumes.length,
      volatility: this.calculateVolatility(closes),
      timestamp: new Date(),
    };
  }

  calculateVolatility(prices) {
    if (prices.length < 2) return 0;

    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;

    return Math.sqrt(variance * 252); // Annualized volatility
  }

  getMarketSummary() {
    const summary = {};

    for (const [key, data] of Object.entries(this.indicators)) {
      summary[key] = {
        ...data,
        bufferSize: this.dataBuffer.get(key)?.length || 0,
      };
    }

    return summary;
  }
}

// ===================== Performance Tracker =====================
class PerformanceTracker {
  constructor() {
    this.metrics = {
      predictions: { total: 0, success: 0, failed: 0 },
      requests: { total: 0, mt5: 0, api: 0 },
      latency: { predictions: [], requests: [] },
      errors: [],
    };

    this.startTime = Date.now();
  }

  trackPrediction(success, latency) {
    this.metrics.predictions.total++;

    if (success) {
      this.metrics.predictions.success++;
    } else {
      this.metrics.predictions.failed++;
    }

    this.metrics.latency.predictions.push(latency);

    // Keep only last 100 latency measurements
    if (this.metrics.latency.predictions.length > 100) {
      this.metrics.latency.predictions.shift();
    }
  }

  trackRequest(source, latency) {
    this.metrics.requests.total++;
    this.metrics.requests[source]++;

    this.metrics.latency.requests.push(latency);

    if (this.metrics.latency.requests.length > 100) {
      this.metrics.latency.requests.shift();
    }
  }

  trackError(error) {
    this.metrics.errors.push({
      timestamp: new Date(),
      error: error.message || error,
      stack: error.stack,
    });

    // Keep only last 50 errors
    if (this.metrics.errors.length > 50) {
      this.metrics.errors.shift();
    }
  }

  getReport() {
    const avgPredictionLatency =
      this.metrics.latency.predictions.length > 0
        ? this.metrics.latency.predictions.reduce((a, b) => a + b, 0) /
          this.metrics.latency.predictions.length
        : 0;

    const avgRequestLatency =
      this.metrics.latency.requests.length > 0
        ? this.metrics.latency.requests.reduce((a, b) => a + b, 0) /
          this.metrics.latency.requests.length
        : 0;

    return {
      uptime: Date.now() - this.startTime,
      predictions: {
        ...this.metrics.predictions,
        successRate:
          this.metrics.predictions.total > 0
            ? (this.metrics.predictions.success /
                this.metrics.predictions.total) *
              100
            : 0,
        avgLatency: avgPredictionLatency,
      },
      requests: {
        ...this.metrics.requests,
        avgLatency: avgRequestLatency,
      },
      errors: this.metrics.errors.length,
      lastError:
        this.metrics.errors.length > 0
          ? this.metrics.errors[this.metrics.errors.length - 1]
          : null,
    };
  }
}

// ===================== Express Server Setup =====================
class TradingBridgeServer {
  constructor() {
    this.app = express();
    this.httpServer = null;
    this.wsServer = null;

    // Initialize components
    this.redisManager = new RedisManager();
    this.aiWorker = new AIWorkerClient();
    this.marketData = new MarketDataManager();
    this.performance = new PerformanceTracker();

    // WebSocket clients
    this.wsClients = new Set();

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
  }

  setupMiddleware() {
    // Security
    this.app.use(helmet());

    // CORS
    this.app.use(
      cors({
        origin: process.env.ALLOWED_ORIGINS?.split(",") || "*",
        credentials: true,
      })
    );

    // Compression
    this.app.use(compression());
    // Fix MT5 NULL-byte bug + JSON parsing
    this.app.use((req, res, next) => {
      let raw = "";

      req.on("data", (chunk) => {
        raw += chunk.toString();
      });

      req.on("end", () => {
        // Bersihkan NULL bytes dari MT5
        raw = raw.replace(/\u0000/g, "");
        req.rawBody = raw;

        // Parse JSON kalau kontennya JSON
        if (req.headers["content-type"]?.includes("application/json") && raw) {
          try {
            req.body = JSON.parse(raw);
          } catch (err) {
            return res.status(400).json({ error: "Invalid JSON" });
          }
        }

        next();
      });

      req.on("error", () => {
        return res.status(400).json({ error: "Bad request stream" });
      });
    });

    // Hapus express.json() karena konflik!
    /*
this.app.use(express.json({
    limit: "10mb",
    verify: ...
}));
*/

    this.app.use(express.urlencoded({ extended: true, limit: "10mb" }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: config.RATE_LIMIT_WINDOW,
      max: config.RATE_LIMIT_MAX,
      message: "Too many requests, please try again later.",
    });

    this.app.use("/api/", limiter);

    // Request logging
    this.app.use((req, res, next) => {
      const start = performance.now();

      res.on("finish", () => {
        const duration = performance.now() - start;
        logger.info({
          method: req.method,
          url: req.url,
          status: res.statusCode,
          duration: `${duration.toFixed(2)}ms`,
        });

        this.performance.trackRequest(
          req.headers["x-source"] || "api",
          duration
        );
      });

      next();
    });
  }

  setupRoutes() {
    // Health check
    this.app.get("/health", (req, res) => {
      res.json({
        status: "healthy",
        timestamp: new Date(),
        services: {
          redis: this.redisManager.client.connected,
          aiWorker: this.aiWorker.getStatus(),
          performance: this.performance.getReport(),
        },
      });
    });

    // Prediction endpoint
    this.app.post("/api/predict", async (req, res) => {
        // console.log("=== RAW BODY RECEIVED ===");
        // console.log(req.rawBody);

        // console.log("=== PARSED BODY ===");
        // console.log(req.body);
      const startTime = performance.now();

      try {
        let { symbol, timeframe, candles, confidence_threshold } = req.body;

        if (!symbol || !candles || !Array.isArray(candles)) {
        return res.status(400).json({ error: "Invalid request data" });
        }


        if (candles.length < config.MIN_CANDLES) {
          return res.status(400).json({
            error: `Minimum ${config.MIN_CANDLES} candles required`,
          });
        }

        if (candles.length > config.MAX_CANDLES) {
        candles = candles.slice(candles.length - config.MAX_CANDLES);
        }


        // Check cache
        const cacheKey = `prediction:${symbol}:${timeframe}:${
          candles[candles.length - 1].timestamp
        }`;
        const cached = await this.redisManager.getCachedData(cacheKey);

        if (cached) {
          logger.info("Cache hit for prediction");
          return res.json(cached);
        }

        // Make prediction
        const prediction = await this.aiWorker.predict({
          candles,
          timeframe: timeframe || config.DEFAULT_TIMEFRAME,
          confidence_threshold: confidence_threshold || 0.45,
        });

        // Add metadata
        prediction.symbol = symbol;
        prediction.server_timestamp = new Date();
        prediction.latency = performance.now() - startTime;

        // Cache result
        await this.redisManager.cacheData(cacheKey, prediction);

        // Track performance
        this.performance.trackPrediction(true, prediction.latency);

        // Broadcast to WebSocket clients
        this.broadcastToClients({
          type: "prediction",
          data: prediction,
        });

        res.json(prediction);
      } catch (error) {
        logger.error("Prediction error:", error);
        this.performance.trackPrediction(false, performance.now() - startTime);
        this.performance.trackError(error);

        res.status(500).json({
          error: "Prediction failed",
          message: error.message,
        });
      }
    });

    // Market data endpoint
    this.app.post("/api/market-data", async (req, res) => {
      try {
        const { symbol, timeframe, candle } = req.body;

        // Store candle
        this.marketData.addCandle(symbol, timeframe, candle);

        // Get current market summary
        const summary = this.marketData.indicators[`${symbol}_${timeframe}`];

        res.json({
          success: true,
          summary,
        });
      } catch (error) {
        logger.error("Market data error:", error);
        res.status(500).json({
          error: "Failed to process market data",
        });
      }
    });

    // Batch prediction endpoint
    this.app.post("/api/predict-batch", async (req, res) => {
      try {
        const { predictions } = req.body;

        if (!Array.isArray(predictions)) {
          return res.status(400).json({
            error: "Predictions must be an array",
          });
        }

        const results = await Promise.allSettled(
          predictions.map((p) => this.aiWorker.predict(p))
        );

        const response = results.map((result, index) => {
          if (result.status === "fulfilled") {
            return {
              ...result.value,
              symbol: predictions[index].symbol,
            };
          } else {
            return {
              symbol: predictions[index].symbol,
              error: result.reason.message,
            };
          }
        });

        res.json(response);
      } catch (error) {
        logger.error("Batch prediction error:", error);
        res.status(500).json({
          error: "Batch prediction failed",
        });
      }
    });

    // Performance metrics endpoint
    this.app.get("/api/metrics", (req, res) => {
      res.json({
        performance: this.performance.getReport(),
        market: this.marketData.getMarketSummary(),
        aiWorker: this.aiWorker.getStatus(),
      });
    });

    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        error: "Endpoint not found",
      });
    });

    // Error handler
    this.app.use((err, req, res, next) => {
      logger.error("Unhandled error:", err);
      res.status(500).json({
        error: "Internal server error",
        message:
          process.env.NODE_ENV === "development" ? err.message : undefined,
      });
    });
  }

  setupWebSocket() {
    this.wsServer = new WebSocket.Server({
      port: config.WS_PORT,
      perMessageDeflate: {
        zlibDeflateOptions: {
          chunkSize: 1024,
          memLevel: 7,
          level: 3,
        },
        zlibInflateOptions: {
          chunkSize: 10 * 1024,
        },
      },
    });

    this.wsServer.on("connection", (ws, req) => {
      const clientId = this.generateClientId();

      // Add to clients set
      ws.clientId = clientId;
      this.wsClients.add(ws);

      logger.info(`WebSocket client connected: ${clientId}`);

      // Send welcome message
      ws.send(
        JSON.stringify({
          type: "connected",
          clientId,
          timestamp: new Date(),
        })
      );

      // Handle messages
      ws.on("message", async (message) => {
        try {
          const data = JSON.parse(message);
          await this.handleWebSocketMessage(ws, data);
        } catch (error) {
          logger.error("WebSocket message error:", error);
          ws.send(
            JSON.stringify({
              type: "error",
              error: error.message,
            })
          );
        }
      });

      // Handle close
      ws.on("close", () => {
        this.wsClients.delete(ws);
        logger.info(`WebSocket client disconnected: ${clientId}`);
      });

      // Handle error
      ws.on("error", (error) => {
        logger.error(`WebSocket error for client ${clientId}:`, error);
      });

      // Setup ping/pong
      ws.isAlive = true;
      ws.on("pong", () => {
        ws.isAlive = true;
      });
    });

    // Heartbeat interval
    setInterval(() => {
      this.wsServer.clients.forEach((ws) => {
        if (!ws.isAlive) {
          ws.terminate();
          return;
        }

        ws.isAlive = false;
        ws.ping();
      });
    }, 30000);

    logger.info(`WebSocket server listening on port ${config.WS_PORT}`);
  }

  async handleWebSocketMessage(ws, data) {
    const { type, payload } = data;

    switch (type) {
      case "predict":
        const prediction = await this.aiWorker.predict(payload);
        ws.send(
          JSON.stringify({
            type: "prediction",
            data: prediction,
            requestId: data.requestId,
          })
        );
        break;

      case "subscribe":
        // Subscribe to market data updates
        const { symbol, timeframe } = payload;
        ws.subscriptions = ws.subscriptions || new Set();
        ws.subscriptions.add(`${symbol}_${timeframe}`);
        break;

      case "unsubscribe":
        if (ws.subscriptions) {
          const { symbol, timeframe } = payload;
          ws.subscriptions.delete(`${symbol}_${timeframe}`);
        }
        break;

      case "ping":
        ws.send(JSON.stringify({ type: "pong" }));
        break;

      default:
        ws.send(
          JSON.stringify({
            type: "error",
            error: `Unknown message type: ${type}`,
          })
        );
    }
  }

  broadcastToClients(message) {
    const messageStr = JSON.stringify(message);

    this.wsClients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        // Check if client is subscribed to this data
        if (message.type === "market_update" && client.subscriptions) {
          const key = `${message.data.symbol}_${message.data.timeframe}`;
          if (!client.subscriptions.has(key)) {
            return;
          }
        }

        client.send(messageStr);
      }
    });
  }

  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async start() {
    // Start HTTP server
    this.httpServer = this.app.listen(config.HTTP_PORT, () => {
      logger.info(`🚀 Ultra-Advanced Trading Bridge Server`);
      logger.info(`HTTP server listening on port ${config.HTTP_PORT}`);
      logger.info(`WebSocket server on port ${config.WS_PORT}`);
      logger.info(`Environment: ${process.env.NODE_ENV || "development"}`);
    });

    // Subscribe to Redis channels
    this.redisManager.subscribe("market_updates", (data) => {
      this.broadcastToClients({
        type: "market_update",
        data,
      });
    });

    // Graceful shutdown
    process.on("SIGTERM", () => this.shutdown());
    process.on("SIGINT", () => this.shutdown());
  }

  async shutdown() {
    logger.info("Shutting down server...");

    // Close WebSocket connections
    this.wsClients.forEach((client) => {
      client.send(
        JSON.stringify({
          type: "shutdown",
          message: "Server is shutting down",
        })
      );
      client.close();
    });

    // Close servers
    if (this.wsServer) {
      this.wsServer.close();
    }

    if (this.httpServer) {
      this.httpServer.close();
    }

    // Close Redis connections
    this.redisManager.client.quit();
    this.redisManager.publisher.quit();
    this.redisManager.subscriber.quit();

    logger.info("Server shutdown complete");
    process.exit(0);
  }
}

// ===================== Start Server =====================
const server = new TradingBridgeServer();
server.start().catch((error) => {
  logger.error("Failed to start server:", error);
  process.exit(1);
});

module.exports = TradingBridgeServer;
