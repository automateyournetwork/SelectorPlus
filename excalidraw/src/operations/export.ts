import { z } from 'zod';
import { ExcalidrawResourceNotFoundError } from '../common/errors.js';
import { getDrawing } from './drawings.js';

// Schema for exporting a drawing to JSON
export const ExportToJsonSchema = z.object({
  id: z.string().min(1),
});

// Export a drawing to JSON
export async function exportToJson(id: string): Promise<string> {
  try {
    // Get the drawing
    const drawing = await getDrawing(id);
    
    // Return the JSON content
    return drawing.content;
  } catch (error) {
    if (error instanceof ExcalidrawResourceNotFoundError) {
      throw error;
    }
    throw new Error(`Failed to export drawing to JSON: ${(error as Error).message}`);
  }
}
