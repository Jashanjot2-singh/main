import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { CheckCircle2, XCircle, Lightbulb, TrendingUp } from "lucide-react";
import type { AnalysisData } from "@/pages/Index";

type AnalysisResultsProps = {
  data: AnalysisData;
};

const AnalysisResults = ({ data }: AnalysisResultsProps) => {
  const getScoreColor = (score: number) => {
    if (score >= 75) return "text-success";
    if (score >= 50) return "text-warning";
    return "text-destructive";
  };

  const getScoreBadgeVariant = (score: number): "default" | "secondary" | "destructive" | "outline" => {
    if (score >= 75) return "default";
    if (score >= 50) return "secondary";
    return "destructive";
  };

  return (
    <Card className="shadow-lg animate-in fade-in slide-in-from-bottom-4 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          Match Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Match Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Match Score</span>
            <Badge variant={getScoreBadgeVariant(data.score)} className="text-lg px-3 py-1">
              {data.score}%
            </Badge>
          </div>
          <Progress value={data.score} className="h-3" />
          <p className={`text-sm font-semibold ${getScoreColor(data.score)}`}>
            {data.score >= 75 ? "Excellent Match" : data.score >= 50 ? "Good Match" : "Weak Match"}
          </p>
        </div>

        {/* Strengths */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-success" />
            <h3 className="font-semibold text-foreground">Strengths</h3>
          </div>
          <ul className="space-y-2">
            {data.strengths.map((strength, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                <span className="text-success mt-0.5">•</span>
                <span>{strength}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Gaps */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <XCircle className="w-5 h-5 text-destructive" />
            <h3 className="font-semibold text-foreground">Gaps</h3>
          </div>
          <ul className="space-y-2">
            {data.gaps.map((gap, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                <span className="text-destructive mt-0.5">•</span>
                <span>{gap}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Key Insights */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-accent" />
            <h3 className="font-semibold text-foreground">Key Insights</h3>
          </div>
          <ul className="space-y-2">
            {data.insights.map((insight, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-muted-foreground">
                <span className="text-accent mt-0.5">•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default AnalysisResults;
