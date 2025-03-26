export class ExcalidrawError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ExcalidrawError';
  }
}

export class ExcalidrawValidationError extends ExcalidrawError {
  response?: any;

  constructor(message: string, response?: any) {
    super(message);
    this.name = 'ExcalidrawValidationError';
    this.response = response;
  }
}

export class ExcalidrawResourceNotFoundError extends ExcalidrawError {
  constructor(message: string) {
    super(message);
    this.name = 'ExcalidrawResourceNotFoundError';
  }
}

export class ExcalidrawAuthenticationError extends ExcalidrawError {
  constructor(message: string) {
    super(message);
    this.name = 'ExcalidrawAuthenticationError';
  }
}

export class ExcalidrawPermissionError extends ExcalidrawError {
  constructor(message: string) {
    super(message);
    this.name = 'ExcalidrawPermissionError';
  }
}

export class ExcalidrawRateLimitError extends ExcalidrawError {
  resetAt: Date;

  constructor(message: string, resetAt: Date) {
    super(message);
    this.name = 'ExcalidrawRateLimitError';
    this.resetAt = resetAt;
  }
}

export class ExcalidrawConflictError extends ExcalidrawError {
  constructor(message: string) {
    super(message);
    this.name = 'ExcalidrawConflictError';
  }
}

export function isExcalidrawError(error: any): error is ExcalidrawError {
  return error instanceof ExcalidrawError;
}
