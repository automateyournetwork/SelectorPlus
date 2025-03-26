import { z } from 'zod';
import fs from 'fs/promises';
import path from 'path';
import { ExcalidrawResourceNotFoundError, ExcalidrawValidationError } from '../common/errors.js';

// Define the storage directory for drawings
const STORAGE_DIR = path.join(process.cwd(), 'storage');

// Ensure storage directory exists
async function ensureStorageDir() {
  try {
    await fs.mkdir(STORAGE_DIR, { recursive: true });
  } catch (error) {
    console.error('Failed to create storage directory:', error);
    throw error;
  }
}

// Schema for creating a drawing
export const CreateDrawingSchema = z.object({
  name: z.string().min(1),
  content: z.string().min(1),
});

// Schema for getting a drawing
export const GetDrawingSchema = z.object({
  id: z.string().min(1),
});

// Schema for updating a drawing
export const UpdateDrawingSchema = z.object({
  id: z.string().min(1),
  content: z.string().min(1),
});

// Schema for deleting a drawing
export const DeleteDrawingSchema = z.object({
  id: z.string().min(1),
});

// Schema for listing drawings
export const ListDrawingsSchema = z.object({
  page: z.number().int().min(1).optional().default(1),
  perPage: z.number().int().min(1).max(100).optional().default(10),
});

// Create a new drawing
export async function createDrawing(name: string, content: string): Promise<{ id: string, name: string }> {
  await ensureStorageDir();
  
  // Generate a unique ID for the drawing
  const id = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  
  // Create the drawing file
  const filePath = path.join(STORAGE_DIR, `${id}.json`);
  
  // Save the drawing content
  await fs.writeFile(filePath, content, 'utf-8');
  
  // Create a metadata file for the drawing
  const metadataPath = path.join(STORAGE_DIR, `${id}.meta.json`);
  const metadata = {
    id,
    name,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  
  await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf-8');
  
  return { id, name };
}

// Get a drawing by ID
export async function getDrawing(id: string): Promise<{ id: string, name: string, content: string, metadata: any }> {
  await ensureStorageDir();
  
  // Get the drawing file path
  const filePath = path.join(STORAGE_DIR, `${id}.json`);
  const metadataPath = path.join(STORAGE_DIR, `${id}.meta.json`);
  
  try {
    // Read the drawing content
    const content = await fs.readFile(filePath, 'utf-8');
    
    // Read the metadata
    const metadataStr = await fs.readFile(metadataPath, 'utf-8');
    const metadata = JSON.parse(metadataStr);
    
    return {
      id,
      name: metadata.name,
      content,
      metadata,
    };
  } catch (error) {
    throw new ExcalidrawResourceNotFoundError(`Drawing with ID ${id} not found`);
  }
}

// Update a drawing by ID
export async function updateDrawing(id: string, content: string): Promise<{ id: string, name: string }> {
  await ensureStorageDir();
  
  // Get the drawing file path
  const filePath = path.join(STORAGE_DIR, `${id}.json`);
  const metadataPath = path.join(STORAGE_DIR, `${id}.meta.json`);
  
  try {
    // Check if the drawing exists
    await fs.access(filePath);
    
    // Read the metadata
    const metadataStr = await fs.readFile(metadataPath, 'utf-8');
    const metadata = JSON.parse(metadataStr);
    
    // Update the drawing content
    await fs.writeFile(filePath, content, 'utf-8');
    
    // Update the metadata
    metadata.updatedAt = new Date().toISOString();
    await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf-8');
    
    return { id, name: metadata.name };
  } catch (error) {
    throw new ExcalidrawResourceNotFoundError(`Drawing with ID ${id} not found`);
  }
}

// Delete a drawing by ID
export async function deleteDrawing(id: string): Promise<void> {
  await ensureStorageDir();
  
  // Get the drawing file path
  const filePath = path.join(STORAGE_DIR, `${id}.json`);
  const metadataPath = path.join(STORAGE_DIR, `${id}.meta.json`);
  
  try {
    // Check if the drawing exists
    await fs.access(filePath);
    
    // Delete the drawing file
    await fs.unlink(filePath);
    
    // Delete the metadata file
    await fs.unlink(metadataPath);
  } catch (error) {
    throw new ExcalidrawResourceNotFoundError(`Drawing with ID ${id} not found`);
  }
}

// List all drawings
export async function listDrawings(page: number = 1, perPage: number = 10): Promise<{ drawings: any[], total: number }> {
  await ensureStorageDir();
  
  try {
    // Get all files in the storage directory
    const files = await fs.readdir(STORAGE_DIR);
    
    // Filter metadata files
    const metadataFiles = files.filter(file => file.endsWith('.meta.json'));
    
    // Calculate pagination
    const start = (page - 1) * perPage;
    const end = start + perPage;
    const paginatedFiles = metadataFiles.slice(start, end);
    
    // Read metadata for each drawing
    const drawings = await Promise.all(
      paginatedFiles.map(async (file) => {
        const metadataPath = path.join(STORAGE_DIR, file);
        const metadataStr = await fs.readFile(metadataPath, 'utf-8');
        return JSON.parse(metadataStr);
      })
    );
    
    return {
      drawings,
      total: metadataFiles.length,
    };
  } catch (error) {
    console.error('Failed to list drawings:', error);
    return {
      drawings: [],
      total: 0,
    };
  }
}
