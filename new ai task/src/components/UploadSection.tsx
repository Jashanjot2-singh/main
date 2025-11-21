import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileText, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { AnalysisData } from "@/pages/Index";

type UploadSectionProps = {
  onAnalysisComplete: (data: AnalysisData, resumeText: string, jobDescText: string) => void;
};

const UploadSection = ({ onAnalysisComplete }: UploadSectionProps) => {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescFile, setJobDescFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'resume' | 'jd') => {
    const file = e.target.files?.[0];
    if (file) {
      const validTypes = ['application/pdf', 'text/plain'];
      if (!validTypes.includes(file.type)) {
        toast({
          title: "Invalid file type",
          description: "Please upload a PDF or TXT file",
          variant: "destructive",
        });
        return;
      }
      if (type === 'resume') {
        setResumeFile(file);
      } else {
        setJobDescFile(file);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!resumeFile || !jobDescFile) {
      toast({
        title: "Missing files",
        description: "Please upload both resume and job description",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('jobDescription', jobDescFile);

      const response = await fetch('http://localhost:3000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const result = await response.json();

      onAnalysisComplete(
        {
          score: result.score,
          strengths: result.strengths,
          gaps: result.gaps,
          insights: result.insights,
        },
        result.resumeText,
        result.jobDescriptionText
      );

      toast({
        title: "Analysis complete",
        description: "Resume has been analyzed successfully",
      });
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: "Please make sure the backend server is running",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <Card className="shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="w-5 h-5 text-primary" />
          Upload Documents
        </CardTitle>
        <CardDescription>
          Upload resume and job description to start the analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Resume (PDF/TXT)</label>
          <div className="relative">
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={(e) => handleFileChange(e, 'resume')}
              className="hidden"
              id="resume-upload"
            />
            <label
              htmlFor="resume-upload"
              className="flex items-center gap-2 p-4 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors bg-card"
            >
              <FileText className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {resumeFile ? resumeFile.name : "Choose resume file"}
              </span>
            </label>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Job Description (PDF/TXT)</label>
          <div className="relative">
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={(e) => handleFileChange(e, 'jd')}
              className="hidden"
              id="jd-upload"
            />
            <label
              htmlFor="jd-upload"
              className="flex items-center gap-2 p-4 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors bg-card"
            >
              <FileText className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {jobDescFile ? jobDescFile.name : "Choose job description file"}
              </span>
            </label>
          </div>
        </div>

        <Button
          onClick={handleAnalyze}
          disabled={!resumeFile || !jobDescFile || isAnalyzing}
          className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Upload className="w-4 h-4 mr-2" />
              Analyze Resume
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
};

export default UploadSection;
