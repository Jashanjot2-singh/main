import { useState } from "react";
import UploadSection from "@/components/UploadSection";
import AnalysisResults from "@/components/AnalysisResults";
import ChatInterface from "@/components/ChatInterface";
import { FileText } from "lucide-react";

export type AnalysisData = {
  score: number;
  strengths: string[];
  gaps: string[];
  insights: string[];
};

const Index = () => {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [resumeText, setResumeText] = useState<string>("");
  const [jobDescriptionText, setJobDescriptionText] = useState<string>("");

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-secondary to-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-primary rounded-lg">
              <FileText className="w-6 h-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Resume Screening Tool</h1>
              <p className="text-sm text-muted-foreground">AI-powered candidate matching with RAG</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid gap-8 lg:grid-cols-2">
          <div className="space-y-8">
            <UploadSection
              onAnalysisComplete={(data, resume, jd) => {
                setAnalysisData(data);
                setResumeText(resume);
                setJobDescriptionText(jd);
              }}
            />
            {analysisData && <AnalysisResults data={analysisData} />}
          </div>

          <div className="lg:sticky lg:top-24 lg:self-start">
            <ChatInterface
              resumeText={resumeText}
              jobDescriptionText={jobDescriptionText}
              disabled={!analysisData}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
