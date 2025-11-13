/**
 * API Client
 * Handles HTTP requests with retry logic, rate limiting, and error handling.
 */

class APIClient {
  /**
   * Initialize API client with base configuration.
   * @param {string} baseURL - Base URL for API endpoints
   * @param {number} timeout - Request timeout in milliseconds
   * @param {number} maxRetries - Maximum retry attempts
   */
  constructor(baseURL, timeout = 5000, maxRetries = 3) {
    this.baseURL = baseURL;
    this.timeout = timeout;
    this.maxRetries = maxRetries;
    this.requestQueue = [];
    this.isProcessing = false;

    // Rate limiting: 100 requests per minute
    this.rateLimit = {
      maxRequests: 100,
      windowMs: 60000,
      requests: []
    };
  }

  /**
   * Make GET request with retry logic.
   * @param {string} endpoint - API endpoint path
   * @param {Object} params - Query parameters
   * @returns {Promise<Object>} Response data
   */
  async get(endpoint, params = {}) {
    const url = this._buildURL(endpoint, params);
    return this._requestWithRetry('GET', url);
  }

  /**
   * Make POST request with payload.
   * @param {string} endpoint - API endpoint path
   * @param {Object} data - Request body
   * @returns {Promise<Object>} Response data
   */
  async post(endpoint, data = {}) {
    const url = this._buildURL(endpoint);
    return this._requestWithRetry('POST', url, data);
  }

  /**
   * Make PUT request for updates.
   * @param {string} endpoint - API endpoint path
   * @param {Object} data - Request body
   * @returns {Promise<Object>} Response data
   */
  async put(endpoint, data = {}) {
    const url = this._buildURL(endpoint);
    return this._requestWithRetry('PUT', url, data);
  }

  /**
   * Make DELETE request.
   * @param {string} endpoint - API endpoint path
   * @returns {Promise<Object>} Response data
   */
  async delete(endpoint) {
    const url = this._buildURL(endpoint);
    return this._requestWithRetry('DELETE', url);
  }

  /**
   * Build full URL with query parameters.
   * @private
   */
  _buildURL(endpoint, params = {}) {
    const url = new URL(endpoint, this.baseURL);
    Object.keys(params).forEach(key => {
      url.searchParams.append(key, params[key]);
    });
    return url.toString();
  }

  /**
   * Execute request with exponential backoff retry.
   * @private
   */
  async _requestWithRetry(method, url, data = null, attempt = 1) {
    // Check rate limit
    if (!this._checkRateLimit()) {
      throw new Error('Rate limit exceeded');
    }

    try {
      const options = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this._getAuthToken()}`
        },
        timeout: this.timeout
      };

      if (data && (method === 'POST' || method === 'PUT')) {
        options.body = JSON.stringify(data);
      }

      const response = await fetch(url, options);

      // Handle HTTP errors
      if (!response.ok) {
        throw new HTTPError(response.status, response.statusText);
      }

      return await response.json();

    } catch (error) {
      // Retry on network errors or 5xx status codes
      if (attempt < this.maxRetries && this._shouldRetry(error)) {
        const backoffMs = this._calculateBackoff(attempt);
        console.log(`Retry attempt ${attempt} after ${backoffMs}ms`);
        await this._sleep(backoffMs);
        return this._requestWithRetry(method, url, data, attempt + 1);
      }

      throw error;
    }
  }

  /**
   * Check if request should be retried.
   * @private
   */
  _shouldRetry(error) {
    if (error instanceof HTTPError) {
      // Retry on 5xx server errors and 429 rate limit
      return error.status >= 500 || error.status === 429;
    }
    // Retry on network errors
    return error.name === 'TypeError' || error.name === 'NetworkError';
  }

  /**
   * Calculate exponential backoff delay.
   * @private
   */
  _calculateBackoff(attempt) {
    const baseDelay = 1000; // 1 second
    const maxDelay = 30000; // 30 seconds
    const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);
    // Add jitter: ±20%
    const jitter = delay * 0.2 * (Math.random() * 2 - 1);
    return Math.floor(delay + jitter);
  }

  /**
   * Check rate limit before making request.
   * @private
   */
  _checkRateLimit() {
    const now = Date.now();
    const windowStart = now - this.rateLimit.windowMs;

    // Remove old requests outside window
    this.rateLimit.requests = this.rateLimit.requests.filter(
      timestamp => timestamp > windowStart
    );

    // Check if under limit
    if (this.rateLimit.requests.length >= this.rateLimit.maxRequests) {
      return false;
    }

    // Add current request
    this.rateLimit.requests.push(now);
    return true;
  }

  /**
   * Get authentication token from storage.
   * @private
   */
  _getAuthToken() {
    // TODO: Implement secure token storage
    return localStorage.getItem('auth_token') || '';
  }

  /**
   * Sleep utility for retry delays.
   * @private
   */
  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current rate limit status.
   * @returns {Object} Rate limit info
   */
  getRateLimitStatus() {
    const now = Date.now();
    const windowStart = now - this.rateLimit.windowMs;
    const activeRequests = this.rateLimit.requests.filter(
      timestamp => timestamp > windowStart
    ).length;

    return {
      limit: this.rateLimit.maxRequests,
      remaining: this.rateLimit.maxRequests - activeRequests,
      resetAt: windowStart + this.rateLimit.windowMs
    };
  }
}

/**
 * Custom HTTP Error class.
 */
class HTTPError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'HTTPError';
    this.status = status;
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { APIClient, HTTPError };
}
