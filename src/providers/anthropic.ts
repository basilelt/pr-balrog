import Anthropic from '@anthropic-ai/sdk'
import { z } from 'zod'
import type { AIAdapter, GenerateQuizOptions, Question } from '../types'
import { QuestionSchema } from '../types'
import { buildSystemPrompt, buildUserPrompt } from './prompt'

const ResponseSchema = z.object({
  questions: z.array(QuestionSchema),
})

export class AnthropicAdapter implements AIAdapter {
  private client: Anthropic
  private model: string

  constructor(apiKey: string, model = 'claude-sonnet-4-6') {
    this.client = new Anthropic({ apiKey })
    this.model = model
  }

  async generateQuiz(opts: GenerateQuizOptions): Promise<Question[]> {
    const message = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      system: buildSystemPrompt(),
      messages: [
        {
          role: 'user',
          content: buildUserPrompt(opts.diff, opts.numQuestions, opts.language, opts.additionalPrompt, opts.allowMultiAnswer),
        },
      ],
    })

    const text = message.content
      .filter((b) => b.type === 'text')
      .map((b) => (b as { type: 'text'; text: string }).text)
      .join('')

    return parseQuizResponse(text, opts.numQuestions, opts.allowMultiAnswer)
  }
}

function parseQuizResponse(raw: string, expected: number, allowMultiAnswer = true): Question[] {
  // strip markdown code fences if present
  const cleaned = raw.replace(/^```(?:json)?\n?/m, '').replace(/\n?```$/m, '').trim()

  const parsed = JSON.parse(cleaned)
  const result = ResponseSchema.safeParse(parsed)

  if (!result.success) {
    throw new Error(`AI returned invalid quiz schema: ${result.error.message}`)
  }

  let questions = result.data.questions.slice(0, expected)
  if (questions.length < expected) {
    throw new Error(`AI returned ${questions.length} questions, expected ${expected}`)
  }

  // Enforce single-correct when multi-answer is disabled — the model ignores instructions
  if (!allowMultiAnswer) {
    questions = questions.map((q) => (q.multi || q.correct.length > 1 ? { ...q, correct: [q.correct[0]], multi: false } : q))
  }

  return questions
}

export { parseQuizResponse }
