import { createClient } from '@supabase/supabase-js';
import 'dotenv/config';
import { access, mkdir, writeFile } from 'fs/promises';
import { join } from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLIC_API_KEY!;

async function downloadImage(url: string, outputPath: string) {
    try {
        await access(outputPath);
        // File exists, skip download
        console.log(`Skipping download of ${outputPath}`);
        return;
    } catch {
        // File doesn't exist, proceed with download
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to download image: ${response.statusText}`);
        }
        const buffer = Buffer.from(await response.arrayBuffer());
        await writeFile(outputPath, buffer);
        console.log(`Downloaded thumbnail ${outputPath}`);
    }
}

async function dumpShaders() {
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Fetch all public shaders with their author's username
    const { data: shaders, error } = await supabase
        .from('shader')
        .select('*, profile:profile(username)')
        .eq('visibility', 'public')
        .order('created_at', { ascending: false });

    if (error) {
        console.error('Error fetching shaders:', error);
        return;
    }

    // Create output directory
    const outputDir = join(process.cwd(), 'shaders');
    await mkdir(outputDir, { recursive: true });

    // Process each shader
    for (const shader of shaders) {
        const id = shader.id.toString();

        // Create JSON output
        const parsedBody = JSON.parse(shader.body);
        const { code, ...bodyWithoutCode } = parsedBody;

        const jsonOutput = {
            ...shader,
            thumb_url: undefined,
            author: undefined,
            license: undefined,
            body: bodyWithoutCode
        };

        // Download thumbnail if it exists
        if (shader.thumb_url) {
            try {
                const fullUrl = `${supabaseUrl}/storage/v1/object/public/shaderthumb/${shader.thumb_url}`;
                await downloadImage(fullUrl, join(outputDir, `${id}.jpeg`));
            } catch (error) {
                console.error(`Failed to download thumbnail for shader ${id}:`, error);
            }
        }

        // Create WGSL output
        const wgslCode = JSON.parse(parsedBody.code);

        // Write files
        await writeFile(join(outputDir, `${id}.json`), JSON.stringify(jsonOutput, null, 2));
        await writeFile(join(outputDir, `${id}.wgsl`), wgslCode);

        console.log(`Processed shader ${id}`);
    }

    console.log(`Finished processing ${shaders.length} shaders`);
}

// Run the script
dumpShaders().catch(console.error);
