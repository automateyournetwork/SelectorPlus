export {};

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      EMAIL_HOST: string;
      EMAIL_PORT: number;
      EMAIL_SSL: string;
      EMAIL_ACCOUNT: string;
      EMAIL_PASSWORD: string;
    }
  }
}
