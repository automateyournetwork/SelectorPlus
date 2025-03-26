FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Create storage directory
RUN mkdir -p storage

# Set permissions for storage directory
RUN chmod -R 777 storage

# Expose port (if needed for HTTP server in the future)
# EXPOSE 3000

# Run the server
CMD ["npm", "start"] 