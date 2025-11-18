//+------------------------------------------------------------------+
//|                                   AI_Trading_Bridge_Ultra.mq5   |
//|                     Ultra-Advanced AI Trading System Integration |
//|                                           Version 3.0 Production |
//+------------------------------------------------------------------+
#property copyright "Ultra-Advanced Trading System"
#property link      "https://ultra-trading.ai"
#property version   "3.00"
#property description "Professional AI-powered trading with advanced risk management"

// ===================== Input Parameters =====================
input group "=== AI Server Configuration ==="
input string   ServerURL = "http://127.0.0.1:3000/api/predict";     // AI Server URL
input string   WebSocketURL = "ws://127.0.0.1:8080";                // WebSocket URL
input int      RequestTimeout = 5000;                               // Request Timeout (ms)
input bool     UseWebSocket = true;                                 // Use WebSocket Connection

input group "=== Trading Parameters ==="
input string   TradingSymbol = "XAUUSDm";                           // Trading Symbol (RENAMED)
input ENUM_TIMEFRAMES Timeframe = PERIOD_M15;                      // Timeframe
input double   MinConfidence = 0.45;                               // Minimum Confidence
input int      MinCandles = 200;                                   // Minimum Candles for Analysis
input int      MaxCandles = 500;                                   // Maximum Candles to Send

input group "=== Risk Management ==="
input double   RiskPercentage = 1.0;                               // Risk Per Trade (%)
input double   MaxDailyLoss = 5.0;                                 // Max Daily Loss (%)
input double   MaxPositions = 3;                                   // Maximum Concurrent Positions
input bool     UseDynamicSizing = true;                            // Use Dynamic Position Sizing
input double   MaxLeverage = 10;                                   // Maximum Leverage

input group "=== Order Management ==="
input bool     UseAIStopLoss = true;                              // Use AI-Suggested Stop Loss
input bool     UseAITakeProfit = true;                            // Use AI-Suggested Take Profit
input double   DefaultSLPoints = 200;                             // Default Stop Loss (points)
input double   DefaultTPPoints = 300;                             // Default Take Profit (points)
input int      MaxSlippage = 10;                                  // Maximum Slippage (points)
input int      MagicNumber = 20241113;                            // Magic Number

input group "=== Execution Settings ==="
input bool     EnableTrading = true;                              // Enable Live Trading
input bool     SendNotifications = true;                          // Send Push Notifications
input bool     UseVirtualStops = false;                          // Use Virtual Stop Orders
input int      RetryAttempts = 3;                                // Order Retry Attempts

input group "=== Performance Monitoring ==="
input bool     EnableLogging = true;                             // Enable Detailed Logging
input bool     SavePredictions = true;                           // Save Predictions to File
input int      PerformanceCheckInterval = 3600;                  // Performance Check (seconds)

// ===================== Global Variables =====================
struct PredictionResult {
    string signal;
    double confidence;
    double expected_return;
    string regime;
    string risk_level;
    double stop_loss;
    double take_profit;
    double position_size;
    datetime timestamp;
    string error;
};

struct PerformanceMetrics {
    int total_trades;
    int winning_trades;
    int losing_trades;
    double total_profit;
    double max_drawdown;
    double win_rate;
    double profit_factor;
    double sharpe_ratio;
    datetime last_update;
};

// Global state
PredictionResult last_prediction;
PerformanceMetrics performance;
datetime last_prediction_time = 0;
datetime last_performance_check = 0;
double daily_loss = 0;
datetime current_day = 0;
int active_positions = 0;
bool is_connected = false;
int prediction_errors = 0;

// File handles
int prediction_log_handle = INVALID_HANDLE;
int performance_log_handle = INVALID_HANDLE;

// ===================== Initialization =====================
int OnInit() {
    Print("=================================================");
    Print("Ultra-Advanced AI Trading System v3.0");
    Print("=================================================");
    
    // Validate inputs
    if (!ValidateInputs()) {
        Print("ERROR: Invalid input parameters");
        return INIT_PARAMETERS_INCORRECT;
    }
    
    // Initialize logging
    if (EnableLogging) {
        InitializeLogging();
    }
    
    // Test server connection
    if (!TestServerConnection()) {
        Print("WARNING: Cannot connect to AI server");
        is_connected = false;
    } else {
        is_connected = true;
        Print("✅ Connected to AI server");
    }
    
    // Initialize performance tracking
    InitializePerformance();
    
    // Setup chart
    SetupChart();
    
    // Set timer for periodic tasks
    EventSetTimer(60);
    
    Print("✅ EA initialized successfully");
    Print("Symbol: ", TradingSymbol);
    Print("Timeframe: ", EnumToString(Timeframe));
    Print("Risk per trade: ", RiskPercentage, "%");
    Print("Min confidence: ", MinConfidence);
    
    return INIT_SUCCEEDED;
}

// ===================== Deinitialization =====================
void OnDeinit(const int reason) {
    EventKillTimer();
    
    SavePerformanceReport();
    
    if (prediction_log_handle != INVALID_HANDLE) {
        FileClose(prediction_log_handle);
    }
    if (performance_log_handle != INVALID_HANDLE) {
        FileClose(performance_log_handle);
    }
    
    Comment("");
    
    string reason_text = GetDeinitReasonText(reason);
    Print("EA stopped: ", reason_text);
}

// ===================== Main Trading Logic =====================
void OnTick() {
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(TradingSymbol, Timeframe, 0);
    if (current_bar_time == last_bar_time) {
        return;
    }
    last_bar_time = current_bar_time;
    
    Print("=== NEW BAR DETECTED ===");
    Print("Time: ", TimeToString(current_bar_time, TIME_DATE|TIME_MINUTES));
    
    UpdateDailyLoss();
    
    if (!CheckRiskLimits()) {
        Comment("Risk limits exceeded - Trading paused");
        return;
    }
    
    if (!is_connected) {
        if (!TestServerConnection()) {
            Comment("Disconnected from AI server");
            return;
        }
        is_connected = true;
    }
    
    MqlRates rates[];
    int copied = CopyRates(TradingSymbol, Timeframe, 0, MinCandles, rates);
    Print("Candle copied: ", copied, " / MinCandles: ", MinCandles);

    
    if (copied < MinCandles) {
        Print("Insufficient data: ", copied, " candles");
        return;
    }
    
    string json_data = PrepareMarketData(rates, copied);
    
    PredictionResult prediction = GetAIPrediction(json_data);
    Print("=== JSON READY ===");
    Print("Sending ", copied, " candles to server");
    if (prediction.error != "") {
        HandlePredictionError(prediction.error);
        return;
    }
    
    if (SavePredictions) {
        LogPrediction(prediction);
    }
    
    UpdateDisplay(prediction);
    
    if (EnableTrading && prediction.confidence >= MinConfidence) {
        ExecuteTradingSignal(prediction);
    }
    
    ManagePositions(prediction);
}

// ===================== Timer Event =====================
void OnTimer() {
    if (!is_connected) {
        TestServerConnection();
    }
    
    datetime current_time = TimeCurrent();
    if (current_time - last_performance_check > PerformanceCheckInterval) {
        UpdatePerformanceMetrics();
        last_performance_check = current_time;
    }
    
    CheckStuckOrders();
}

// ===================== AI Server Communication =====================
PredictionResult GetAIPrediction(string market_data) {
    PredictionResult result;
    result.error = "";
    
    datetime current_time = TimeCurrent();
    if (current_time - last_prediction_time < 5) {
        result.error = "Rate limited";
        return result;
    }
    last_prediction_time = current_time;
    
    string headers = "Content-Type: application/json\r\n";
    headers += "X-Source: MT5\r\n";
    
    string request_body = "{\"symbol\":\"" + TradingSymbol + 
                         "\",\"timeframe\":\"" + TimeframeToString(Timeframe) + 
                         "\",\"candles\":" + market_data + 
                         ",\"confidence_threshold\":" + DoubleToString(MinConfidence, 2) + "}";
    
    char post_data[];
    StringToCharArray(request_body, post_data, 0, StringLen(request_body));
    
    char server_response[];
    string response_headers;
    
    ResetLastError();
    int response_code = WebRequest(
        "POST",
        ServerURL,
        headers,
        RequestTimeout,
        post_data,
        server_response,
        response_headers
    );
    
    Print("=== SERVER RESPONSE ===");
    Print("HTTP Code: ", response_code);
    
    if (response_code == -1) {
        int error = GetLastError();
        result.error = "WebRequest failed: " + IntegerToString(error);
        
        if (error == 4014) {
            result.error += " (URL not allowed in terminal settings)";
        }
        
        return result;
    }
    
    if (response_code != 200) {
        result.error = "Server error: HTTP " + IntegerToString(response_code);
        return result;
    }
    
    string response_str = CharArrayToString(server_response);
    if (!ParsePredictionResponse(response_str, result)) {
        result.error = "Failed to parse server response";
    }
    
    return result;
}

bool ParsePredictionResponse(string json, PredictionResult &result) {
    int signal_start = StringFind(json, "\"signal\":\"") + 10;
    int signal_end = StringFind(json, "\"", signal_start);
    if (signal_start > 9 && signal_end > signal_start) {
        result.signal = StringSubstr(json, signal_start, signal_end - signal_start);
    }
    
    int conf_start = StringFind(json, "\"confidence\":") + 13;
    int conf_end = StringFind(json, ",", conf_start);
    if (conf_start > 12 && conf_end > conf_start) {
        result.confidence = StringToDouble(StringSubstr(json, conf_start, conf_end - conf_start));
    }
    
    int ret_start = StringFind(json, "\"expected_return\":") + 18;
    int ret_end = StringFind(json, ",", ret_start);
    if (ret_start > 17 && ret_end > ret_start) {
        result.expected_return = StringToDouble(StringSubstr(json, ret_start, ret_end - ret_start));
    }
    
    int regime_start = StringFind(json, "\"regime\":\"") + 10;
    int regime_end = StringFind(json, "\"", regime_start);
    if (regime_start > 9 && regime_end > regime_start) {
        result.regime = StringSubstr(json, regime_start, regime_end - regime_start);
    }
    
    int risk_start = StringFind(json, "\"risk_level\":\"") + 14;
    int risk_end = StringFind(json, "\"", risk_start);
    if (risk_start > 13 && risk_end > risk_start) {
        result.risk_level = StringSubstr(json, risk_start, risk_end - risk_start);
    }
    
    if (exec_start > 0) {
        int sl_start = StringFind(json, "\"stop_loss\":") + 12;
        int sl_end = StringFind(json, ",", sl_start);
        if (sl_start > 11 && sl_end > sl_start) {
            result.stop_loss = StringToDouble(StringSubstr(json, sl_start, sl_end - sl_start));
        }
        
        int tp_start = StringFind(json, "\"take_profit\":") + 14;
        int tp_end = StringFind(json, ",", tp_start);
        if (tp_start > 13 && tp_end > tp_start) {
            result.take_profit = StringToDouble(StringSubstr(json, tp_start, tp_end - tp_start));
        }
        
        int size_start = StringFind(json, "\"recommended_size\":", exec_start) + 19;
        int size_end = StringFind(json, ",", size_start);
        if (size_start > 18 && size_end > size_start) {
            result.position_size = StringToDouble(StringSubstr(json, size_start, size_end - size_start));
        }
    }
    
    result.timestamp = TimeCurrent();
    
    return (result.signal != "");
}

string PrepareMarketData(MqlRates &rates[], int count) {
    int data_count = MathMin(count, MaxCandles);
    int start_index = count - data_count;
    
    string json = "[";
    
    for (int i = start_index; i < count; i++) {
        if (i > start_index) json += ",";
        
        json += "{\"time\":\"" + TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES) + 
                "\",\"open\":" + DoubleToString(rates[i].open, 5) +
                ",\"high\":" + DoubleToString(rates[i].high, 5) +
                ",\"low\":" + DoubleToString(rates[i].low, 5) +
                ",\"close\":" + DoubleToString(rates[i].close, 5) +
                ",\"volume\":" + IntegerToString((int)rates[i].tick_volume) + "}";
    }
    
    json += "]";
    
    return json;
}

// ===================== Trade Execution =====================
void ExecuteTradingSignal(PredictionResult &prediction) {
    if (active_positions >= MaxPositions) {
        Print("Max positions reached");
        return;
    }
    
    if (prediction.signal == "HOLD") {
        return;
    }
    
    double lot_size = CalculatePositionSize(prediction);
    if (lot_size <= 0) {
        Print("Invalid position size calculated");
        return;
    }
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.symbol = TradingSymbol;
    request.volume = lot_size;
    request.magic = MagicNumber;
    request.deviation = MaxSlippage;
    
    if (prediction.signal == "BUY") {
        request.action = TRADE_ACTION_DEAL;
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(TradingSymbol, SYMBOL_ASK);
    } else if (prediction.signal == "SELL") {
        request.action = TRADE_ACTION_DEAL;
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(TradingSymbol, SYMBOL_BID);
    } else {
        return;
    }
    
    if (UseAIStopLoss && prediction.stop_loss > 0) {
        request.sl = prediction.stop_loss;
    } else {
        double sl_distance = DefaultSLPoints * SymbolInfoDouble(TradingSymbol, SYMBOL_POINT);
        request.sl = (prediction.signal == "BUY") ? 
            request.price - sl_distance : request.price + sl_distance;
    }
    
    if (UseAITakeProfit && prediction.take_profit > 0) {
        request.tp = prediction.take_profit;
    } else {
        double tp_distance = DefaultTPPoints * SymbolInfoDouble(TradingSymbol, SYMBOL_POINT);
        request.tp = (prediction.signal == "BUY") ? 
            request.price + tp_distance : request.price - tp_distance;
    }
    
    request.comment = "AI|" + prediction.signal + "|Conf:" + DoubleToString(prediction.confidence, 2) + 
                     "|Regime:" + prediction.regime + "|Risk:" + prediction.risk_level;
    
    bool order_placed = false;
    for (int attempt = 0; attempt < RetryAttempts; attempt++) {
        if (OrderSend(request, result)) {
            order_placed = true;
            break;
        }
        
        Print("Order failed, attempt ", attempt + 1, ": ", result.comment);
        Sleep(1000);
    }
    
    if (order_placed) {
        active_positions++;
        
        LogTrade(request, result, prediction);
        
        if (SendNotifications) {
            SendTradeNotification(prediction.signal, lot_size, prediction.confidence);
        }
        
        Print("✅ Order executed: ", prediction.signal, 
              " Lot: ", lot_size, 
              " Confidence: ", prediction.confidence);
    } else {
        Print("❌ Failed to execute order after ", RetryAttempts, " attempts");
    }
}

double CalculatePositionSize(PredictionResult &prediction) {
    double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = account_balance * (RiskPercentage / 100.0);
    
    double point = SymbolInfoDouble(TradingSymbol, SYMBOL_POINT);
    double tick_value = SymbolInfoDouble(TradingSymbol, SYMBOL_TRADE_TICK_VALUE);
    double min_lot = SymbolInfoDouble(TradingSymbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(TradingSymbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(TradingSymbol, SYMBOL_VOLUME_STEP);
    
    double sl_distance;
    if (UseAIStopLoss && prediction.stop_loss > 0) {
        double current_price = (prediction.signal == "BUY") ? 
            SymbolInfoDouble(TradingSymbol, SYMBOL_ASK) : SymbolInfoDouble(TradingSymbol, SYMBOL_BID);
        sl_distance = MathAbs(current_price - prediction.stop_loss) / point;
    } else {
        sl_distance = DefaultSLPoints;
    }
    
    double lot_size = risk_amount / (sl_distance * tick_value);
    
    if (UseDynamicSizing) {
        double confidence_multiplier = MathPow(prediction.confidence, 2);
        
        double risk_multiplier = 1.0;
        if (prediction.risk_level == "HIGH") {
            risk_multiplier = 0.5;
        } else if (prediction.risk_level == "MEDIUM") {
            risk_multiplier = 0.75;
        }
        
        double regime_multiplier = 1.0;
        if (prediction.regime == "range") {
            regime_multiplier = 0.8;
        } else if (prediction.regime == "strong_trend") {
            regime_multiplier = 1.2;
        }
        
        if (prediction.position_size > 0) {
            double ai_size = account_balance * prediction.position_size;
            lot_size = (lot_size + ai_size) / 2;
        }
        
        lot_size *= confidence_multiplier * risk_multiplier * regime_multiplier;
    }
    
    lot_size = MathFloor(lot_size / lot_step) * lot_step;
    
    lot_size = MathMax(min_lot, MathMin(lot_size, max_lot));
    
    double required_margin = lot_size * SymbolInfoDouble(TradingSymbol, SYMBOL_MARGIN_INITIAL);
    double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    
    if (required_margin > free_margin) {
        lot_size = (free_margin * 0.9) / SymbolInfoDouble(TradingSymbol, SYMBOL_MARGIN_INITIAL);
        lot_size = MathFloor(lot_size / lot_step) * lot_step;
    }
    
    return lot_size;
}

// ===================== Position Management =====================
void ManagePositions(PredictionResult &prediction) {
    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        if (!PositionSelectByTicket(PositionGetTicket(i))) continue;
        
        if (PositionGetString(POSITION_SYMBOL) != TradingSymbol) continue;
        if (PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        
        long position_type = PositionGetInteger(POSITION_TYPE);
        
        if (prediction.confidence >= MinConfidence * 1.2) {
            if ((position_type == POSITION_TYPE_BUY && prediction.signal == "SELL") ||
                (position_type == POSITION_TYPE_SELL && prediction.signal == "BUY")) {
                
                ClosePosition(PositionGetTicket(i), "AI reversal signal");
            }
        }
        
        if (UseVirtualStops) {
            TrailStopLoss(PositionGetTicket(i));
        }
    }
}

void ClosePosition(ulong ticket, string reason) {
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    if (!PositionSelectByTicket(ticket)) return;
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = PositionGetString(POSITION_SYMBOL);
    request.volume = PositionGetDouble(POSITION_VOLUME);
    request.position = ticket;
    request.magic = MagicNumber;
    
    if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
    } else {
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
    }
    
    request.deviation = MaxSlippage;
    request.comment = "Close: " + reason;
    
    if (OrderSend(request, result)) {
        active_positions--;
        Print("Position closed: ", reason);
    }
}

void TrailStopLoss(ulong ticket) {
    if (!PositionSelectByTicket(ticket)) return;
    
    double current_sl = PositionGetDouble(POSITION_SL);
    double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
    double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    
    if ((PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && current_price > open_price) ||
        (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && current_price < open_price)) {
        
        double point = SymbolInfoDouble(TradingSymbol, SYMBOL_POINT);
        double trail_distance = DefaultSLPoints * point;
        double new_sl;
        
        if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
            new_sl = current_price - trail_distance;
            if (new_sl > current_sl) {
                ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
        } else {
            new_sl = current_price + trail_distance;
            if (new_sl < current_sl || current_sl == 0) {
                ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
        }
    }
}

void ModifyPosition(ulong ticket, double sl, double tp) {
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_SLTP;
    request.position = ticket;
    request.sl = sl;
    request.tp = tp;
    request.magic = MagicNumber;
    
    if (!OrderSend(request, result)) {
        Print("Failed to modify position: ", result.comment);
    }
}

// ===================== Risk Management =====================
bool CheckRiskLimits() {
    if (daily_loss >= AccountInfoDouble(ACCOUNT_BALANCE) * (MaxDailyLoss / 100.0)) {
        Print("Daily loss limit reached");
        return false;
    }
    
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (balance - equity) / balance * 100;
    
    if (drawdown > MaxDailyLoss) {
        Print("Drawdown limit exceeded: ", drawdown, "%");
        return false;
    }
    
    double margin_level = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
    if (margin_level > 0 && margin_level < 200) {
        Print("Low margin level: ", margin_level, "%");
        return false;
    }
    
    return true;
}

void UpdateDailyLoss() {
    datetime current = TimeCurrent();
    datetime day_start = current - (current % 86400);
    
    if (day_start != current_day) {
        current_day = day_start;
        daily_loss = 0;
        
        CalculateDailyPerformance();
    }
    
    double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    static double day_start_balance = current_balance;
    
    if (current_day == day_start) {
        daily_loss = MathMax(0, day_start_balance - current_balance);
    }
}

// ===================== Performance Tracking =====================
void InitializePerformance() {
    performance.total_trades = 0;
    performance.winning_trades = 0;
    performance.losing_trades = 0;
    performance.total_profit = 0;
    performance.max_drawdown = 0;
    performance.win_rate = 0;
    performance.profit_factor = 0;
    performance.sharpe_ratio = 0;
    performance.last_update = TimeCurrent();
}

void UpdatePerformanceMetrics() {
    int total = 0, wins = 0, losses = 0;
    double gross_profit = 0, gross_loss = 0;
    
    datetime from = TimeCurrent() - 30 * 86400;
    HistorySelect(from, TimeCurrent());
    
    for (int i = 0; i < HistoryDealsTotal(); i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if (ticket == 0) continue;
        
        if (HistoryDealGetString(ticket, DEAL_SYMBOL) != TradingSymbol) continue;
        if (HistoryDealGetInteger(ticket, DEAL_MAGIC) != MagicNumber) continue;
        if (HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
        
        double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
        total++;
        
        if (profit > 0) {
            wins++;
            gross_profit += profit;
        } else if (profit < 0) {
            losses++;
            gross_loss += MathAbs(profit);
        }
    }
    
    performance.total_trades = total;
    performance.winning_trades = wins;
    performance.losing_trades = losses;
    performance.total_profit = gross_profit - gross_loss;
    
    if (total > 0) {
        performance.win_rate = (double)wins / total * 100;
    }
    
    if (gross_loss > 0) {
        performance.profit_factor = gross_profit / gross_loss;
    }
    
    performance.last_update = TimeCurrent();
    
    if (total > 1) {
        double returns[];
        ArrayResize(returns, total);
        int index = 0;
        
        for (int i = 0; i < HistoryDealsTotal() && index < total; i++) {
            ulong ticket = HistoryDealGetTicket(i);
            if (HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                returns[index++] = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            }
        }
        
        double mean = 0, std_dev = 0;
        for (int i = 0; i < total; i++) mean += returns[i];
        mean /= total;
        
        for (int i = 0; i < total; i++) {
            std_dev += MathPow(returns[i] - mean, 2);
        }
        std_dev = MathSqrt(std_dev / total);
        
        if (std_dev > 0) {
            performance.sharpe_ratio = mean / std_dev * MathSqrt(252);
        }
    }
}

void CalculateDailyPerformance() {
    datetime yesterday_start = current_day - 86400;
    datetime yesterday_end = current_day - 1;
    
    HistorySelect(yesterday_start, yesterday_end);
    
    double daily_profit = 0;
    int daily_trades = 0;
    
    for (int i = 0; i < HistoryDealsTotal(); i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if (HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber) {
            daily_profit += HistoryDealGetDouble(ticket, DEAL_PROFIT);
            if (HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                daily_trades++;
            }
        }
    }
    
    if (EnableLogging && performance_log_handle != INVALID_HANDLE) {
        string log_entry = TimeToString(yesterday_start, TIME_DATE) + 
                          ",Daily Summary,Trades:" + IntegerToString(daily_trades) + 
                          ",Profit:" + DoubleToString(daily_profit, 2);
        FileWrite(performance_log_handle, log_entry);
    }
}

// ===================== Logging and Monitoring =====================
void InitializeLogging() {
    string timestamp = TimeToString(TimeCurrent(), TIME_DATE);
    StringReplace(timestamp, ".", "_");
    
    if (SavePredictions) {
        string prediction_filename = "AI_Predictions_" + TradingSymbol + "_" + timestamp + ".csv";
        prediction_log_handle = FileOpen(prediction_filename, FILE_WRITE|FILE_CSV|FILE_COMMON);
        
        if (prediction_log_handle != INVALID_HANDLE) {
            FileWrite(prediction_log_handle, 
                "Timestamp", "Signal", "Confidence", "ExpectedReturn", 
                "Regime", "RiskLevel", "StopLoss", "TakeProfit");
        }
    }
    
    string performance_filename = "AI_Performance_" + TradingSymbol + "_" + timestamp + ".csv";
    performance_log_handle = FileOpen(performance_filename, FILE_WRITE|FILE_CSV|FILE_COMMON);
    
    if (performance_log_handle != INVALID_HANDLE) {
        FileWrite(performance_log_handle, 
            "Timestamp", "Event", "Details", "Balance", "Equity", "Profit");
    }
}

void LogPrediction(PredictionResult &prediction) {
    if (prediction_log_handle == INVALID_HANDLE) return;
    
    FileWrite(prediction_log_handle,
        TimeToString(prediction.timestamp, TIME_DATE | TIME_SECONDS),
        prediction.signal,
        DoubleToString(prediction.confidence, 4),
        DoubleToString(prediction.expected_return, 6),
        prediction.regime,
        prediction.risk_level,
        DoubleToString(prediction.stop_loss, 5),
        DoubleToString(prediction.take_profit, 5)
    );
    
    FileFlush(prediction_log_handle);
}

void LogTrade(MqlTradeRequest &request, MqlTradeResult &result, PredictionResult &prediction) {
    if (performance_log_handle == INVALID_HANDLE) return;
    
    string trade_details = "Order:" + IntegerToString(result.order) + 
                          ",Type:" + ((request.type == ORDER_TYPE_BUY) ? "BUY" : "SELL") +
                          ",Lot:" + DoubleToString(request.volume, 2) +
                          ",Price:" + DoubleToString(request.price, 5) +
                          ",SL:" + DoubleToString(request.sl, 5) +
                          ",TP:" + DoubleToString(request.tp, 5) +
                          ",Conf:" + DoubleToString(prediction.confidence, 2);
    
    FileWrite(performance_log_handle,
        TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
        "TRADE_OPENED",
        trade_details,
        AccountInfoDouble(ACCOUNT_BALANCE),
        AccountInfoDouble(ACCOUNT_EQUITY),
        0
    );
    
    FileFlush(performance_log_handle);
}

void SavePerformanceReport() {
    if (performance_log_handle == INVALID_HANDLE) return;
    
    UpdatePerformanceMetrics();
    
    string report = "FINAL_REPORT: Trades:" + IntegerToString(performance.total_trades) +
                   ",Wins:" + IntegerToString(performance.winning_trades) +
                   ",Losses:" + IntegerToString(performance.losing_trades) +
                   ",WinRate:" + DoubleToString(performance.win_rate, 1) + "%" +
                   ",ProfitFactor:" + DoubleToString(performance.profit_factor, 2) +
                   ",Sharpe:" + DoubleToString(performance.sharpe_ratio, 2) +
                   ",TotalProfit:" + DoubleToString(performance.total_profit, 2);
    
    FileWrite(performance_log_handle,
        TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
        "PERFORMANCE_SUMMARY",
        report,
        AccountInfoDouble(ACCOUNT_BALANCE),
        AccountInfoDouble(ACCOUNT_EQUITY),
        performance.total_profit
    );
    
    FileFlush(performance_log_handle);
}

// ===================== Display and Notifications =====================
void UpdateDisplay(PredictionResult &prediction) {
    string display = "╔══════════════════════════════════════╗\n";
    display += "║  ULTRA-ADVANCED AI TRADING SYSTEM   ║\n";
    display += "╠══════════════════════════════════════╣\n";
    
    display += "║ Status: " + (is_connected ? "✅ Connected" : "❌ Disconnected") + "\n";
    
    display += "║ Signal: " + prediction.signal + " (" + 
               DoubleToString(prediction.confidence * 100, 1) + "%)\n";
    display += "║ Expected Return: " + 
               DoubleToString(prediction.expected_return * 100, 4) + "%\n";
    display += "║ Regime: " + prediction.regime + " | Risk: " + prediction.risk_level + "\n";
    
    display += "╠══════════════════════════════════════╣\n";
    display += "║ Today's P/L: " + DoubleToString(-daily_loss, 2) + "\n";
    display += "║ Win Rate: " + DoubleToString(performance.win_rate, 1) + "% (" + 
               IntegerToString(performance.winning_trades) + "/" + 
               IntegerToString(performance.total_trades) + ")\n";
    display += "║ Profit Factor: " + DoubleToString(performance.profit_factor, 2) + "\n";
    display += "║ Sharpe Ratio: " + DoubleToString(performance.sharpe_ratio, 2) + "\n";
    
    display += "╠══════════════════════════════════════╣\n";
    display += "║ Balance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + "\n";
    display += "║ Equity: " + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + "\n";
    display += "║ Margin: " + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_LEVEL), 2) + "%\n";
    display += "║ Positions: " + IntegerToString(active_positions) + "/" + 
               IntegerToString((int)MaxPositions) + "\n";
    
    display += "╚══════════════════════════════════════╝";
    
    Comment(display);
}

void SendTradeNotification(string signal, double lot, double confidence) {
    if (!SendNotifications) return;
    
    string message = "AI Trade: " + signal + " " + TradingSymbol + 
                    "\nLot: " + DoubleToString(lot, 2) +
                    "\nConfidence: " + DoubleToString(confidence * 100, 1) + "%" +
                    "\nBalance: " + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2);
    
    SendNotification(message);
}

// ===================== Utility Functions =====================
bool ValidateInputs() {
    if (RiskPercentage <= 0 || RiskPercentage > 10) {
        Print("Invalid risk percentage: ", RiskPercentage);
        return false;
    }
    
    if (MinConfidence < 0 || MinConfidence > 1) {
        Print("Invalid minimum confidence: ", MinConfidence);
        return false;
    }
    
    if (MinCandles < 20) {
        Print("Minimum candles too low: ", MinCandles);
        return false;
    }
    
    return true;
}

bool TestServerConnection() {
    uchar empty_data[];        // GET tidak pakai body
    uchar server_response[];   // buffer response
    string response_headers;   // output headers

    // Build health URL
    int health_url_end = StringFind(ServerURL, "/api");
    string health_url =
        (health_url_end > 0)
        ? StringSubstr(ServerURL, 0, health_url_end) + "/health"
        : ServerURL + "/health";

    // Send GET request
    ResetLastError();
    int response_code = WebRequest(
        "GET",
        health_url,
        "",               // no headers needed
        5000,             // timeout
        empty_data,       // GET = no body, but must pass empty uchar[]
        server_response,
        response_headers
    );

    // Return success only for 200 OK
    return (response_code == 200);
}

void SetupChart() {
    ChartSetInteger(0, CHART_SHOW_GRID, false);
    ChartSetInteger(0, CHART_SHOW_VOLUMES, CHART_VOLUME_TICK);
    ChartSetInteger(0, CHART_COLOR_BACKGROUND, clrBlack);
    ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrWhite);
    ChartSetInteger(0, CHART_COLOR_CANDLE_BULL, clrLime);
    ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR, clrRed);
}

void CheckStuckOrders() {
    for (int i = OrdersTotal() - 1; i >= 0; i--) {
        ulong ticket = OrderGetTicket(i);
        if (ticket > 0) {
            if (OrderGetString(ORDER_SYMBOL) == TradingSymbol && 
                OrderGetInteger(ORDER_MAGIC) == MagicNumber) {
                
                datetime order_time = (datetime)OrderGetInteger(ORDER_TIME_SETUP);
                if (TimeCurrent() - order_time > 300) {
                    Print("Cancelling stuck order: ", ticket);
                    
                    MqlTradeRequest request = {};
                    MqlTradeResult result = {};
                    
                    request.action = TRADE_ACTION_REMOVE;
                    request.order = ticket;
                    
                    OrderSend(request, result);
                }
            }
        }
    }
}

void HandlePredictionError(string error) {
    prediction_errors++;
    
    Print("Prediction error: ", error);
    
    if (prediction_errors > 5) {
        is_connected = false;
        Print("Too many prediction errors, marking as disconnected");
    }
}

string TimeframeToString(ENUM_TIMEFRAMES tf) {
    switch(tf) {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        case PERIOD_W1:  return "W1";
        case PERIOD_MN1: return "MN1";
        default: return "M15";
    }
}

string GetDeinitReasonText(int reason) {
    switch(reason) {
        case REASON_PROGRAM: return "Program terminated";
        case REASON_REMOVE: return "Program removed from chart";
        case REASON_RECOMPILE: return "Program recompiled";
        case REASON_CHARTCHANGE: return "Chart symbol or period changed";
        case REASON_CHARTCLOSE: return "Chart closed";
        case REASON_PARAMETERS: return "Input parameters changed";
        case REASON_ACCOUNT: return "Account changed";
        default: return "Unknown reason";
    }
}

//+------------------------------------------------------------------+