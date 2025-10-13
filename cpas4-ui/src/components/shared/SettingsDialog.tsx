import { useState, useRef } from 'react';
import { Image, Upload, X } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Slider } from '../ui/slider';
import { useUIStore } from '../../stores/uiStore';

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const {
    backgroundImage,
    backgroundOpacity,
    setBackgroundImage,
    setBackgroundOpacity,
  } = useUIStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewImage, setPreviewImage] = useState<string | undefined>(backgroundImage);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setPreviewImage(result);
        setBackgroundImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveBackground = () => {
    setPreviewImage(undefined);
    setBackgroundImage(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Customize your CPAS4 interface
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Background Image Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Image className="h-4 w-4 text-primary" />
                <h3 className="font-medium text-sm">Background Image</h3>
              </div>
              {previewImage && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRemoveBackground}
                  className="h-8 text-xs"
                >
                  <X className="h-3 w-3 mr-1" />
                  Remove
                </Button>
              )}
            </div>

            {/* Image Preview */}
            {previewImage && (
              <div className="relative w-full h-32 rounded-md overflow-hidden border border-border">
                <img
                  src={previewImage}
                  alt="Background preview"
                  className="w-full h-full object-cover"
                  style={{ opacity: backgroundOpacity }}
                />
              </div>
            )}

            {/* Upload Button */}
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="w-full"
              >
                <Upload className="h-4 w-4 mr-2" />
                {previewImage ? 'Change Background' : 'Upload Background'}
              </Button>
            </div>

            <p className="text-xs text-muted-foreground">
              Supported formats: JPG, PNG, GIF. Recommended size: 1920x1080px
            </p>
          </div>

          {/* Background Opacity Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-sm">Background Opacity</h3>
              <span className="text-sm text-muted-foreground">
                {Math.round(backgroundOpacity * 100)}%
              </span>
            </div>

            <Slider
              value={[backgroundOpacity]}
              onValueChange={(value) => setBackgroundOpacity(value[0])}
              min={0}
              max={1}
              step={0.05}
              className="w-full"
            />

            <p className="text-xs text-muted-foreground">
              Adjust the opacity of the background image overlay
            </p>
          </div>

          {/* API Configuration Section */}
          <div className="space-y-3 border-t border-border pt-4">
            <h3 className="font-medium text-sm">API Configuration</h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between px-3 py-2 rounded-md bg-accent/50">
                <span className="text-sm">Backend URL</span>
                <span className="text-xs text-muted-foreground font-mono">
                  localhost:8000
                </span>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

